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
// CHECK:STDOUT:   %AddWith.type.b35: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.b35 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.302: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self.236: %AddWith.type.302 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.054: type = symbolic_binding_type Self, 1, %Self.236 [symbolic]
// CHECK:STDOUT:   %pattern_type.7a8: type = pattern_type %Self.binding.as_type.054 [symbolic]
// CHECK:STDOUT:   %AddWith.Op.type.8bf: type = fn_type @AddWith.Op, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %AddWith.Op.349: %AddWith.Op.type.8bf = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.fb4: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.c22: %AddWith.assoc_type.fb4 = assoc_entity element0, @AddWith.%AddWith.Op.decl [symbolic]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.596: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.654: %As.type.596 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.854: type = symbolic_binding_type Self, 1, %Self.654 [symbolic]
// CHECK:STDOUT:   %pattern_type.054: type = pattern_type %Self.binding.as_type.854 [symbolic]
// CHECK:STDOUT:   %pattern_type.e68: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %As.Convert.type.7fa: type = fn_type @As.Convert, @As(%T) [symbolic]
// CHECK:STDOUT:   %As.Convert.1e3: %As.Convert.type.7fa = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type.335: type = assoc_entity_type @As, @As(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.b00: %As.assoc_type.335 = assoc_entity element0, @As.%As.Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.841: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.f50: %ImplicitAs.type.841 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.31f: type = symbolic_binding_type Self, 1, %Self.f50 [symbolic]
// CHECK:STDOUT:   %pattern_type.4e2: type = pattern_type %Self.binding.as_type.31f [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.178: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.1a0: %ImplicitAs.Convert.type.178 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.88c: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.7d3: %ImplicitAs.assoc_type.88c = assoc_entity element0, @ImplicitAs.%ImplicitAs.Convert.decl [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %AddWith.type.f1c: type = facet_type <@AddWith, @AddWith(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.203: %AddWith.type.f1c = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.Op.type.efc: type = fn_type @AddWith.Op, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %AddWith.Op.a62: %AddWith.Op.type.efc = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.a65: type = assoc_entity_type @AddWith, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.cb9: %AddWith.assoc_type.a65 = assoc_entity element0, @AddWith.%AddWith.Op.decl [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness file.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.956: type = pattern_type %i32.builtin [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.type: type = fn_type @i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op: %i32.builtin.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.f1c = facet_value %i32.builtin, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %As.type.cf8: type = facet_type <@As, @As(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.21a: %As.type.cf8 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.Convert.type.c0d: type = fn_type @As.Convert, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %As.Convert.713: %As.Convert.type.c0d = struct_value () [concrete]
// CHECK:STDOUT:   %As.assoc_type.b9b: type = assoc_entity_type @As, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.925: %As.assoc_type.b9b = assoc_entity element0, @As.%As.Convert.decl [concrete]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness file.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.cf8 = facet_value Core.IntLiteral, (%As.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.adc: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.3bf: %ImplicitAs.type.adc = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.752: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.fcc: %ImplicitAs.Convert.type.752 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.1cf: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.7c4: %ImplicitAs.assoc_type.1cf = assoc_entity element0, @ImplicitAs.%ImplicitAs.Convert.decl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a3a: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc27 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7c0: %ImplicitAs.type.adc = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a3a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0d7: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Self.f3e: %ImplicitAs.type.0d7 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.60e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.c73: %ImplicitAs.Convert.type.60e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.014: type = assoc_entity_type @ImplicitAs, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc0.2ad: %ImplicitAs.assoc_type.014 = assoc_entity element0, @ImplicitAs.%ImplicitAs.Convert.decl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.942: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc31 [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert: %i32.builtin.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.4f3: %ImplicitAs.type.0d7 = facet_value %i32.builtin, (%ImplicitAs.impl_witness.942) [concrete]
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
// CHECK:STDOUT:   %AddWith.decl: %AddWith.type.b35 = interface_decl @AddWith [concrete = constants.%AddWith.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc7_19.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_19.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc11_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc11_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
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
// CHECK:STDOUT:     %AddWith.ref: %AddWith.type.b35 = name_ref AddWith, file.%AddWith.decl [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:     %int_32.loc19_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call.loc19_21: init type = call constants.%Int(%int_32.loc19_21) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc19_24.1: type = value_of_initializer %Int.call.loc19_21 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc19_24.2: type = converted %Int.call.loc19_21, %.loc19_24.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %AddWith.type: type = facet_type <@AddWith, @AddWith(constants.%i32.builtin)> [concrete = constants.%AddWith.type.f1c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddWith.impl_witness_table = impl_witness_table (@i32.builtin.as.AddWith.impl.%i32.builtin.as.AddWith.impl.Op.decl), @i32.builtin.as.AddWith.impl [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness %AddWith.impl_witness_table [concrete = constants.%AddWith.impl_witness]
// CHECK:STDOUT:   impl_decl @Core.IntLiteral.as.As.impl [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_17.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_17.2: type = converted %IntLiteral.call, %.loc23_17.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %As.ref: %As.type.b51 = name_ref As, file.%As.decl [concrete = constants.%As.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc23_28.1: type = value_of_initializer %Int.call [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc23_28.2: type = converted %Int.call, %.loc23_28.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%i32.builtin)> [concrete = constants.%As.type.cf8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (@Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.decl), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness %As.impl_witness_table [concrete = constants.%As.impl_witness]
// CHECK:STDOUT:   impl_decl @Core.IntLiteral.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc27_17.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc27_17.2: type = converted %IntLiteral.call, %.loc27_17.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.96f = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc27_36.1: type = value_of_initializer %Int.call [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc27_36.2: type = converted %Int.call, %.loc27_36.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32.builtin)> [concrete = constants.%ImplicitAs.type.adc]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc27 = impl_witness_table (@Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.decl), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc27: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc27 [concrete = constants.%ImplicitAs.impl_witness.a3a]
// CHECK:STDOUT:   impl_decl @i32.builtin.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call: init type = call constants.%Int(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc31_6.1: type = value_of_initializer %Int.call [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc31_6.2: type = converted %Int.call, %.loc31_6.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.96f = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_36.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_36.2: type = converted %IntLiteral.call, %.loc31_36.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete = constants.%ImplicitAs.type.0d7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc31 = impl_witness_table (@i32.builtin.as.ImplicitAs.impl.%i32.builtin.as.ImplicitAs.impl.Convert.decl), @i32.builtin.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc31: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc31 [concrete = constants.%ImplicitAs.impl_witness.942]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(%T.loc7_19.2: type) {
// CHECK:STDOUT:   %T.loc7_19.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_19.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T.loc7_19.1)> [symbolic = %AddWith.type (constants.%AddWith.type.302)]
// CHECK:STDOUT:   %Self.2: @AddWith.%AddWith.type (%AddWith.type.302) = symbolic_binding Self, 1 [symbolic = %Self.2 (constants.%Self.236)]
// CHECK:STDOUT:   %AddWith.Op.type: type = fn_type @AddWith.Op, @AddWith(%T.loc7_19.1) [symbolic = %AddWith.Op.type (constants.%AddWith.Op.type.8bf)]
// CHECK:STDOUT:   %AddWith.Op: @AddWith.%AddWith.Op.type (%AddWith.Op.type.8bf) = struct_value () [symbolic = %AddWith.Op (constants.%AddWith.Op.349)]
// CHECK:STDOUT:   %AddWith.assoc_type: type = assoc_entity_type @AddWith, @AddWith(%T.loc7_19.1) [symbolic = %AddWith.assoc_type (constants.%AddWith.assoc_type.fb4)]
// CHECK:STDOUT:   %assoc0.loc8_41.2: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.fb4) = assoc_entity element0, %AddWith.Op.decl [symbolic = %assoc0.loc8_41.2 (constants.%assoc0.c22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @AddWith.%AddWith.type (%AddWith.type.302) = symbolic_binding Self, 1 [symbolic = %Self.2 (constants.%Self.236)]
// CHECK:STDOUT:     %AddWith.Op.decl: @AddWith.%AddWith.Op.type (%AddWith.Op.type.8bf) = fn_decl @AddWith.Op [symbolic = @AddWith.%AddWith.Op (constants.%AddWith.Op.349)] {
// CHECK:STDOUT:       %self.patt: @AddWith.Op.%pattern_type (%pattern_type.7a8) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @AddWith.Op.%pattern_type (%pattern_type.7a8) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %other.patt: @AddWith.Op.%pattern_type (%pattern_type.7a8) = value_binding_pattern other [concrete]
// CHECK:STDOUT:       %other.param_patt: @AddWith.Op.%pattern_type (%pattern_type.7a8) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @AddWith.Op.%pattern_type (%pattern_type.7a8) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @AddWith.Op.%pattern_type (%pattern_type.7a8) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc8_37.1: @AddWith.Op.%AddWith.type (%AddWith.type.302) = specific_constant @AddWith.%Self.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.236)]
// CHECK:STDOUT:       %Self.ref.loc8_37: @AddWith.Op.%AddWith.type (%AddWith.type.302) = name_ref Self, %.loc8_37.1 [symbolic = %Self (constants.%Self.236)]
// CHECK:STDOUT:       %Self.as_type.loc8_37: type = facet_access_type %Self.ref.loc8_37 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.054)]
// CHECK:STDOUT:       %.loc8_37.2: type = converted %Self.ref.loc8_37, %Self.as_type.loc8_37 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.054)]
// CHECK:STDOUT:       %self.param: @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.054) = value_param call_param0
// CHECK:STDOUT:       %.loc8_15.1: type = splice_block %.loc8_15.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.054)] {
// CHECK:STDOUT:         %.loc8_15.2: @AddWith.Op.%AddWith.type (%AddWith.type.302) = specific_constant @AddWith.%Self.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.236)]
// CHECK:STDOUT:         %Self.ref.loc8_15: @AddWith.Op.%AddWith.type (%AddWith.type.302) = name_ref Self, %.loc8_15.2 [symbolic = %Self (constants.%Self.236)]
// CHECK:STDOUT:         %Self.as_type.loc8_15: type = facet_access_type %Self.ref.loc8_15 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.054)]
// CHECK:STDOUT:         %.loc8_15.3: type = converted %Self.ref.loc8_15, %Self.as_type.loc8_15 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.054)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.054) = value_binding self, %self.param
// CHECK:STDOUT:       %other.param: @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.054) = value_param call_param1
// CHECK:STDOUT:       %.loc8_28.1: type = splice_block %.loc8_28.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.054)] {
// CHECK:STDOUT:         %.loc8_28.2: @AddWith.Op.%AddWith.type (%AddWith.type.302) = specific_constant @AddWith.%Self.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.236)]
// CHECK:STDOUT:         %Self.ref.loc8_28: @AddWith.Op.%AddWith.type (%AddWith.type.302) = name_ref Self, %.loc8_28.2 [symbolic = %Self (constants.%Self.236)]
// CHECK:STDOUT:         %Self.as_type.loc8_28: type = facet_access_type %Self.ref.loc8_28 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.054)]
// CHECK:STDOUT:         %.loc8_28.3: type = converted %Self.ref.loc8_28, %Self.as_type.loc8_28 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.054)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %other: @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.054) = value_binding other, %other.param
// CHECK:STDOUT:       %return.param: ref @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.054) = out_param call_param2
// CHECK:STDOUT:       %return: ref @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.054) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc8_41.1: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.fb4) = assoc_entity element0, %AddWith.Op.decl [symbolic = %assoc0.loc8_41.2 (constants.%assoc0.c22)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Op = %assoc0.loc8_41.1
// CHECK:STDOUT:     witness = (%AddWith.Op.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%T.loc11_14.2: type) {
// CHECK:STDOUT:   %T.loc11_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc11_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T.loc11_14.1)> [symbolic = %As.type (constants.%As.type.596)]
// CHECK:STDOUT:   %Self.2: @As.%As.type (%As.type.596) = symbolic_binding Self, 1 [symbolic = %Self.2 (constants.%Self.654)]
// CHECK:STDOUT:   %As.Convert.type: type = fn_type @As.Convert, @As(%T.loc11_14.1) [symbolic = %As.Convert.type (constants.%As.Convert.type.7fa)]
// CHECK:STDOUT:   %As.Convert: @As.%As.Convert.type (%As.Convert.type.7fa) = struct_value () [symbolic = %As.Convert (constants.%As.Convert.1e3)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%T.loc11_14.1) [symbolic = %As.assoc_type (constants.%As.assoc_type.335)]
// CHECK:STDOUT:   %assoc0.loc12_32.2: @As.%As.assoc_type (%As.assoc_type.335) = assoc_entity element0, %As.Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.b00)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.596) = symbolic_binding Self, 1 [symbolic = %Self.2 (constants.%Self.654)]
// CHECK:STDOUT:     %As.Convert.decl: @As.%As.Convert.type (%As.Convert.type.7fa) = fn_decl @As.Convert [symbolic = @As.%As.Convert (constants.%As.Convert.1e3)] {
// CHECK:STDOUT:       %self.patt: @As.Convert.%pattern_type.loc12_14 (%pattern_type.054) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @As.Convert.%pattern_type.loc12_14 (%pattern_type.054) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @As.Convert.%pattern_type.loc12_28 (%pattern_type.e68) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @As.Convert.%pattern_type.loc12_28 (%pattern_type.e68) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @As.%T.loc11_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @As.Convert.%Self.binding.as_type (%Self.binding.as_type.854) = value_param call_param0
// CHECK:STDOUT:       %.loc12_20.1: type = splice_block %.loc12_20.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.854)] {
// CHECK:STDOUT:         %.loc12_20.2: @As.Convert.%As.type (%As.type.596) = specific_constant @As.%Self.1, @As(constants.%T) [symbolic = %Self (constants.%Self.654)]
// CHECK:STDOUT:         %Self.ref: @As.Convert.%As.type (%As.type.596) = name_ref Self, %.loc12_20.2 [symbolic = %Self (constants.%Self.654)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.854)]
// CHECK:STDOUT:         %.loc12_20.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.854)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @As.Convert.%Self.binding.as_type (%Self.binding.as_type.854) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @As.Convert.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @As.Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc12_32.1: @As.%As.assoc_type (%As.assoc_type.335) = assoc_entity element0, %As.Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.b00)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc12_32.1
// CHECK:STDOUT:     witness = (%As.Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc15_22.2: type) {
// CHECK:STDOUT:   %T.loc15_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc15_22.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.841)]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.841) = symbolic_binding Self, 1 [symbolic = %Self.2 (constants.%Self.f50)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%T.loc15_22.1) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type.178)]
// CHECK:STDOUT:   %ImplicitAs.Convert: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.178) = struct_value () [symbolic = %ImplicitAs.Convert (constants.%ImplicitAs.Convert.1a0)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T.loc15_22.1) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.88c)]
// CHECK:STDOUT:   %assoc0.loc16_32.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.88c) = assoc_entity element0, %ImplicitAs.Convert.decl [symbolic = %assoc0.loc16_32.2 (constants.%assoc0.7d3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.841) = symbolic_binding Self, 1 [symbolic = %Self.2 (constants.%Self.f50)]
// CHECK:STDOUT:     %ImplicitAs.Convert.decl: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.178) = fn_decl @ImplicitAs.Convert [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.1a0)] {
// CHECK:STDOUT:       %self.patt: @ImplicitAs.Convert.%pattern_type.loc16_14 (%pattern_type.4e2) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @ImplicitAs.Convert.%pattern_type.loc16_14 (%pattern_type.4e2) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @ImplicitAs.Convert.%pattern_type.loc16_28 (%pattern_type.e68) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @ImplicitAs.Convert.%pattern_type.loc16_28 (%pattern_type.e68) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc15_22.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.Convert.%Self.binding.as_type (%Self.binding.as_type.31f) = value_param call_param0
// CHECK:STDOUT:       %.loc16_20.1: type = splice_block %.loc16_20.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.31f)] {
// CHECK:STDOUT:         %.loc16_20.2: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.841) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self.f50)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.841) = name_ref Self, %.loc16_20.2 [symbolic = %Self (constants.%Self.f50)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.31f)]
// CHECK:STDOUT:         %.loc16_20.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.31f)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.Convert.%Self.binding.as_type (%Self.binding.as_type.31f) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ImplicitAs.Convert.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ImplicitAs.Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc16_32.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.88c) = assoc_entity element0, %ImplicitAs.Convert.decl [symbolic = %assoc0.loc16_32.2 (constants.%assoc0.7d3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = %assoc0.loc16_32.1
// CHECK:STDOUT:     witness = (%ImplicitAs.Convert.decl)
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %i32.builtin.as.AddWith.impl.Op.decl
// CHECK:STDOUT:   witness = file.%AddWith.impl_witness
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Core.IntLiteral.as.As.impl.Convert.decl
// CHECK:STDOUT:   witness = file.%As.impl_witness
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Core.IntLiteral.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc27
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .IntLiteral = <poisoned>
// CHECK:STDOUT:   .Convert = %i32.builtin.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.Op(@AddWith.%T.loc7_19.2: type, @AddWith.%Self.1: @AddWith.%AddWith.type (%AddWith.type.302)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.302)]
// CHECK:STDOUT:   %Self: @AddWith.Op.%AddWith.type (%AddWith.type.302) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.236)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.054)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.7a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.054), %other.param: @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.054)) -> @AddWith.Op.%Self.binding.as_type (%Self.binding.as_type.054);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @As.Convert(@As.%T.loc11_14.2: type, @As.%Self.1: @As.%As.type (%As.type.596)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.596)]
// CHECK:STDOUT:   %Self: @As.Convert.%As.type (%As.type.596) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.654)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.854)]
// CHECK:STDOUT:   %pattern_type.loc12_14: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc12_14 (constants.%pattern_type.054)]
// CHECK:STDOUT:   %pattern_type.loc12_28: type = pattern_type %T [symbolic = %pattern_type.loc12_28 (constants.%pattern_type.e68)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @As.Convert.%Self.binding.as_type (%Self.binding.as_type.854)) -> @As.Convert.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.Convert(@ImplicitAs.%T.loc15_22.2: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.841)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.841)]
// CHECK:STDOUT:   %Self: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.841) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.f50)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.31f)]
// CHECK:STDOUT:   %pattern_type.loc16_14: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc16_14 (constants.%pattern_type.4e2)]
// CHECK:STDOUT:   %pattern_type.loc16_28: type = pattern_type %T [symbolic = %pattern_type.loc16_28 (constants.%pattern_type.e68)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ImplicitAs.Convert.%Self.binding.as_type (%Self.binding.as_type.31f)) -> @ImplicitAs.Convert.%T (%T);
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
// CHECK:STDOUT: specific @AddWith.Op(constants.%T, constants.%Self.236) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.302
// CHECK:STDOUT:   %Self => constants.%Self.236
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.054
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7a8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.Convert(constants.%T, constants.%Self.654) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.596
// CHECK:STDOUT:   %Self => constants.%Self.654
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.854
// CHECK:STDOUT:   %pattern_type.loc12_14 => constants.%pattern_type.054
// CHECK:STDOUT:   %pattern_type.loc12_28 => constants.%pattern_type.e68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(constants.%T, constants.%Self.f50) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.841
// CHECK:STDOUT:   %Self => constants.%Self.f50
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.31f
// CHECK:STDOUT:   %pattern_type.loc16_14 => constants.%pattern_type.4e2
// CHECK:STDOUT:   %pattern_type.loc16_28 => constants.%pattern_type.e68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc7_19.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.f1c
// CHECK:STDOUT:   %Self.2 => constants.%Self.203
// CHECK:STDOUT:   %AddWith.Op.type => constants.%AddWith.Op.type.efc
// CHECK:STDOUT:   %AddWith.Op => constants.%AddWith.Op.a62
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.a65
// CHECK:STDOUT:   %assoc0.loc8_41.2 => constants.%assoc0.cb9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.Op(constants.%i32.builtin, constants.%AddWith.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.f1c
// CHECK:STDOUT:   %Self => constants.%AddWith.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%i32.builtin
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.956
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc11_14.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.cf8
// CHECK:STDOUT:   %Self.2 => constants.%Self.21a
// CHECK:STDOUT:   %As.Convert.type => constants.%As.Convert.type.c0d
// CHECK:STDOUT:   %As.Convert => constants.%As.Convert.713
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.b9b
// CHECK:STDOUT:   %assoc0.loc12_32.2 => constants.%assoc0.925
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.Convert(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.cf8
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
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.adc
// CHECK:STDOUT:   %Self.2 => constants.%Self.3bf
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.752
// CHECK:STDOUT:   %ImplicitAs.Convert => constants.%ImplicitAs.Convert.fcc
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.1cf
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.7c4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(constants.%i32.builtin, constants.%ImplicitAs.facet.7c0) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.adc
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.7c0
// CHECK:STDOUT:   %Self.binding.as_type => Core.IntLiteral
// CHECK:STDOUT:   %pattern_type.loc16_14 => constants.%pattern_type.dc0
// CHECK:STDOUT:   %pattern_type.loc16_28 => constants.%pattern_type.956
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc15_22.1 => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.0d7
// CHECK:STDOUT:   %Self.2 => constants.%Self.f3e
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.60e
// CHECK:STDOUT:   %ImplicitAs.Convert => constants.%ImplicitAs.Convert.c73
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.014
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.2ad
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(Core.IntLiteral, constants.%ImplicitAs.facet.4f3) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.0d7
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.4f3
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
// CHECK:STDOUT:   %As.type.2c2: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.aa9: %As.type.2c2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.Convert.type.169: type = fn_type @As.Convert, @As(%T) [symbolic]
// CHECK:STDOUT:   %As.Convert.70f: %As.Convert.type.169 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.e68: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.d87: type = symbolic_binding_type Self, 1, %Self.aa9 [symbolic]
// CHECK:STDOUT:   %pattern_type.8ac: type = pattern_type %Self.binding.as_type.d87 [symbolic]
// CHECK:STDOUT:   %As.assoc_type.10e: type = assoc_entity_type @As, @As(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.40b: %As.assoc_type.10e = assoc_entity element0, imports.%Core.import_ref.7a4 [symbolic]
// CHECK:STDOUT:   %As.type.a29: type = facet_type <@As, @As(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.a6c: %As.type.a29 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.Convert.type.378: type = fn_type @As.Convert, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %As.Convert.e51: %As.Convert.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %As.assoc_type.bc2: type = assoc_entity_type @As, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.124: %As.assoc_type.bc2 = assoc_entity element0, imports.%Core.import_ref.7a4 [concrete]
// CHECK:STDOUT:   %assoc0.964: %As.assoc_type.10e = assoc_entity element0, imports.%Core.import_ref.048 [symbolic]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness imports.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a29 = facet_value Core.IntLiteral, (%As.impl_witness) [concrete]
// CHECK:STDOUT:   %.8dd: type = fn_type_with_self_type %As.Convert.type.378, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.e07: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.f38: %i32.builtin = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.5db: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_2.5a1: %i32.builtin = int_value 2 [concrete]
// CHECK:STDOUT:   %AddWith.type.e05: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.e05 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.51e: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self.895: %AddWith.type.51e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.Op.type.e78: type = fn_type @AddWith.Op, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %AddWith.Op.e03: %AddWith.Op.type.e78 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.b1f: type = symbolic_binding_type Self, 1, %Self.895 [symbolic]
// CHECK:STDOUT:   %pattern_type.086: type = pattern_type %Self.binding.as_type.b1f [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.dc1: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.03b: %AddWith.assoc_type.dc1 = assoc_entity element0, imports.%Core.import_ref.4ac [symbolic]
// CHECK:STDOUT:   %AddWith.type.2a3: type = facet_type <@AddWith, @AddWith(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.558: %AddWith.type.2a3 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.Op.type.0b7: type = fn_type @AddWith.Op, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %AddWith.Op.9d6: %AddWith.Op.type.0b7 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.dff: type = assoc_entity_type @AddWith, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.a24: %AddWith.assoc_type.dff = assoc_entity element0, imports.%Core.import_ref.4ac [concrete]
// CHECK:STDOUT:   %assoc0.941: %AddWith.assoc_type.dc1 = assoc_entity element0, imports.%Core.import_ref.91e [symbolic]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness imports.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.2a3 = facet_value %i32.builtin, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.f31: type = fn_type_with_self_type %AddWith.Op.type.0b7, %AddWith.facet [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.type: type = fn_type @i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op: %i32.builtin.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.bound.9a7: <bound method> = bound_method %int_1.f38, %i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_3.a0f: %i32.builtin = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2d9: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.4f1: %ImplicitAs.type.2d9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4c8: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.e9f: %ImplicitAs.Convert.type.4c8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.ab0: type = symbolic_binding_type Self, 1, %Self.4f1 [symbolic]
// CHECK:STDOUT:   %pattern_type.c40: type = pattern_type %Self.binding.as_type.ab0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.8b5: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.f64: %ImplicitAs.assoc_type.8b5 = assoc_entity element0, imports.%Core.import_ref.b91295.1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7a9: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Self.886: %ImplicitAs.type.7a9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.0e2: %ImplicitAs.Convert.type.71e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.959: type = assoc_entity_type @ImplicitAs, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc0.5cf: %ImplicitAs.assoc_type.959 = assoc_entity element0, imports.%Core.import_ref.b91295.1 [concrete]
// CHECK:STDOUT:   %assoc0.7b6: %ImplicitAs.assoc_type.8b5 = assoc_entity element0, imports.%Core.import_ref.d11 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.873: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.418: %ImplicitAs.type.873 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.059: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.4d7: %ImplicitAs.Convert.type.059 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.398: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.5eb: %ImplicitAs.assoc_type.398 = assoc_entity element0, imports.%Core.import_ref.b91295.2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb8: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.16d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7e7: %ImplicitAs.type.7a9 = facet_value %i32.builtin, (%ImplicitAs.impl_witness.fb8) [concrete]
// CHECK:STDOUT:   %.f56: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.7e7 [concrete]
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
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c15: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b46 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2fc: %ImplicitAs.type.873 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c15) [concrete]
// CHECK:STDOUT:   %.63c: type = fn_type_with_self_type %ImplicitAs.Convert.type.059, %ImplicitAs.facet.2fc [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.efcd44.1: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.375 = import_ref Core//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.471: @As.%As.assoc_type (%As.assoc_type.10e) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%assoc0 (constants.%assoc0.964)]
// CHECK:STDOUT:   %Core.Convert.924 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.efcd44.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.1a4: @As.%As.type (%As.type.2c2) = import_ref Core//default, inst{{[0-9A-F]+}} [no loc], loaded [symbolic = @As.%Self (constants.%Self.aa9)]
// CHECK:STDOUT:   %Core.import_ref.7a4: @As.%As.Convert.type (%As.Convert.type.169) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%As.Convert (constants.%As.Convert.70f)]
// CHECK:STDOUT:   %Core.import_ref.048 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.931: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%As.impl_witness]
// CHECK:STDOUT:   %Core.import_ref.8721d7.1: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.a7f: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%As.type.a29]
// CHECK:STDOUT:   %Core.import_ref.a36: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.a36), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.AddWith: %AddWith.type.e05 = import_ref Core//default, AddWith, loaded [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:   %Core.import_ref.efcd44.3: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.833 = import_ref Core//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.d7c: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.dc1) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%assoc0 (constants.%assoc0.941)]
// CHECK:STDOUT:   %Core.Op = import_ref Core//default, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.efcd44.4: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.54a: @AddWith.%AddWith.type (%AddWith.type.51e) = import_ref Core//default, inst{{[0-9A-F]+}} [no loc], loaded [symbolic = @AddWith.%Self (constants.%Self.895)]
// CHECK:STDOUT:   %Core.import_ref.4ac: @AddWith.%AddWith.Op.type (%AddWith.Op.type.e78) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%AddWith.Op (constants.%AddWith.Op.e03)]
// CHECK:STDOUT:   %Core.import_ref.91e = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.8af: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%AddWith.impl_witness]
// CHECK:STDOUT:   %Core.import_ref.c8c7cd.1: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %Core.import_ref.af3: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%AddWith.type.2a3]
// CHECK:STDOUT:   %Core.import_ref.980: %i32.builtin.as.AddWith.impl.Op.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:   %AddWith.impl_witness_table = impl_witness_table (%Core.import_ref.980), @i32.builtin.as.AddWith.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.efcd44.5: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.d6f = import_ref Core//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.b9c: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.8b5) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.7b6)]
// CHECK:STDOUT:   %Core.Convert.acf = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.efcd44.6: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.4ff: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2d9) = import_ref Core//default, inst{{[0-9A-F]+}} [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.4f1)]
// CHECK:STDOUT:   %Core.import_ref.b91295.1: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.4c8) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.e9f)]
// CHECK:STDOUT:   %Core.import_ref.d11 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.8a9: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.impl_witness.c15]
// CHECK:STDOUT:   %Core.import_ref.8721d7.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.64f: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.type.873]
// CHECK:STDOUT:   %Core.import_ref.b91295.2: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.4c8) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.e9f)]
// CHECK:STDOUT:   %Core.import_ref.614: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.impl_witness.fb8]
// CHECK:STDOUT:   %Core.import_ref.c8c7cd.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %Core.import_ref.d7b: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.type.7a9]
// CHECK:STDOUT:   %Core.import_ref.4ae: %i32.builtin.as.ImplicitAs.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.16d = impl_witness_table (%Core.import_ref.4ae), @i32.builtin.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.564: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b46 = impl_witness_table (%Core.import_ref.564), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %impl.elem0.loc4_24: %.8dd = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_24: <bound method> = bound_method %int_1, %impl.elem0.loc4_24 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.e07]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call.loc4_24: init %i32.builtin = call %bound_method.loc4_24(%int_1) [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_24.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_24 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_24.2: %i32.builtin = converted %int_1, %.loc4_24.1 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %int_32.loc4_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call.loc4_40: init type = call constants.%Int(%int_32.loc4_40) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_40.1: type = value_of_initializer %Int.call.loc4_40 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_40.2: type = converted %Int.call.loc4_40, %.loc4_40.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_37: %.8dd = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_37: <bound method> = bound_method %int_2, %impl.elem0.loc4_37 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.5db]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call.loc4_37: init %i32.builtin = call %bound_method.loc4_37(%int_2) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_37.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_37 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_37.2: %i32.builtin = converted %int_2, %.loc4_37.1 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %impl.elem0.loc4_32.1: %.f31 = impl_witness_access constants.%AddWith.impl_witness, element0 [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:     %bound_method.loc4_32.1: <bound method> = bound_method %.loc4_24.2, %impl.elem0.loc4_32.1 [concrete = constants.%i32.builtin.as.AddWith.impl.Op.bound.9a7]
// CHECK:STDOUT:     %i32.builtin.as.AddWith.impl.Op.call: init %i32.builtin = call %bound_method.loc4_32.1(%.loc4_24.2, %.loc4_37.2) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc4_16.1: type = value_of_initializer %Int.call.loc4_16 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_16.2: type = converted %Int.call.loc4_16, %.loc4_16.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_32.2: %.f56 = impl_witness_access constants.%ImplicitAs.impl_witness.fb8, element0 [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT: generic interface @As(imports.%Core.import_ref.efcd44.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.2c2)]
// CHECK:STDOUT:   %Self: @As.%As.type (%As.type.2c2) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.aa9)]
// CHECK:STDOUT:   %As.Convert.type: type = fn_type @As.Convert, @As(%T) [symbolic = %As.Convert.type (constants.%As.Convert.type.169)]
// CHECK:STDOUT:   %As.Convert: @As.%As.Convert.type (%As.Convert.type.169) = struct_value () [symbolic = %As.Convert (constants.%As.Convert.70f)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%T) [symbolic = %As.assoc_type (constants.%As.assoc_type.10e)]
// CHECK:STDOUT:   %assoc0: @As.%As.assoc_type (%As.assoc_type.10e) = assoc_entity element0, imports.%Core.import_ref.7a4 [symbolic = %assoc0 (constants.%assoc0.40b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.375
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.471
// CHECK:STDOUT:     witness = (imports.%Core.Convert.924)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(imports.%Core.import_ref.efcd44.3: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.51e)]
// CHECK:STDOUT:   %Self: @AddWith.%AddWith.type (%AddWith.type.51e) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.895)]
// CHECK:STDOUT:   %AddWith.Op.type: type = fn_type @AddWith.Op, @AddWith(%T) [symbolic = %AddWith.Op.type (constants.%AddWith.Op.type.e78)]
// CHECK:STDOUT:   %AddWith.Op: @AddWith.%AddWith.Op.type (%AddWith.Op.type.e78) = struct_value () [symbolic = %AddWith.Op (constants.%AddWith.Op.e03)]
// CHECK:STDOUT:   %AddWith.assoc_type: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic = %AddWith.assoc_type (constants.%AddWith.assoc_type.dc1)]
// CHECK:STDOUT:   %assoc0: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.dc1) = assoc_entity element0, imports.%Core.import_ref.4ac [symbolic = %assoc0 (constants.%assoc0.03b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.833
// CHECK:STDOUT:     .Op = imports.%Core.import_ref.d7c
// CHECK:STDOUT:     witness = (imports.%Core.Op)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.efcd44.5: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2d9)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2d9) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.4f1)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%T) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type.4c8)]
// CHECK:STDOUT:   %ImplicitAs.Convert: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.4c8) = struct_value () [symbolic = %ImplicitAs.Convert (constants.%ImplicitAs.Convert.e9f)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.8b5)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.8b5) = assoc_entity element0, imports.%Core.import_ref.b91295.1 [symbolic = %assoc0 (constants.%assoc0.f64)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.d6f
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.b9c
// CHECK:STDOUT:     witness = (imports.%Core.Convert.acf)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.As.impl: imports.%Core.import_ref.8721d7.1 as imports.%Core.import_ref.a7f [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.931
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.AddWith.impl: imports.%Core.import_ref.c8c7cd.1 as imports.%Core.import_ref.af3 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.8af
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.ImplicitAs.impl: imports.%Core.import_ref.8721d7.2 as imports.%Core.import_ref.64f [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.8a9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.ImplicitAs.impl: imports.%Core.import_ref.c8c7cd.2 as imports.%Core.import_ref.d7b [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.614
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @As.Convert(imports.%Core.import_ref.efcd44.2: type, imports.%Core.import_ref.1a4: @As.%As.type (%As.type.2c2)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.2c2)]
// CHECK:STDOUT:   %Self: @As.Convert.%As.type (%As.type.2c2) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.aa9)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.d87)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.8ac)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.e68)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Core.IntLiteral.as.As.impl.Convert = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.Op(imports.%Core.import_ref.efcd44.4: type, imports.%Core.import_ref.54a: @AddWith.%AddWith.type (%AddWith.type.51e)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.51e)]
// CHECK:STDOUT:   %Self: @AddWith.Op.%AddWith.type (%AddWith.type.51e) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.895)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.b1f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.086)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.builtin.as.AddWith.impl.Op = "int.sadd" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.Convert(imports.%Core.import_ref.efcd44.6: type, imports.%Core.import_ref.4ff: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2d9)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2d9)]
// CHECK:STDOUT:   %Self: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.2d9) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.4f1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.ab0)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.c40)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.e68)]
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
// CHECK:STDOUT:   %impl.elem0.loc4_58: %.8dd = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_58: <bound method> = bound_method %int_3.loc4_56, %impl.elem0.loc4_58 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.55c]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc4_58: init %i32.builtin = call %bound_method.loc4_58(%int_3.loc4_56) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_58.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_58 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_58.2: %i32.builtin = converted %int_3.loc4_56, %.loc4_58.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %int_4.loc4_69: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %int_32.loc4_74: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %Int.call.loc4_74: init type = call constants.%Int(%int_32.loc4_74) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_74.1: type = value_of_initializer %Int.call.loc4_74 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_74.2: type = converted %Int.call.loc4_74, %.loc4_74.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_71: %.8dd = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_71: <bound method> = bound_method %int_4.loc4_69, %impl.elem0.loc4_71 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.eb9]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc4_71: init %i32.builtin = call %bound_method.loc4_71(%int_4.loc4_69) [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_71.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_71 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_71.2: %i32.builtin = converted %int_4.loc4_69, %.loc4_71.1 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %impl.elem0.loc4_66: %.f31 = impl_witness_access constants.%AddWith.impl_witness, element0 [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc4_66: <bound method> = bound_method %.loc4_58.2, %impl.elem0.loc4_66 [concrete = constants.%i32.builtin.as.AddWith.impl.Op.bound.81b]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.call: init %i32.builtin = call %bound_method.loc4_66(%.loc4_58.2, %.loc4_71.2) [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc4_78.1: %tuple.type = tuple_literal (%int_3.loc4_49, %int_4.loc4_52, %i32.builtin.as.AddWith.impl.Op.call)
// CHECK:STDOUT:   %impl.elem0.loc4_78.1: %.63c = impl_witness_access constants.%ImplicitAs.impl_witness.c15, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_78.1: <bound method> = bound_method %int_3.loc4_49, %impl.elem0.loc4_78.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c88]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.1: init %i32.builtin = call %bound_method.loc4_78.1(%int_3.loc4_49) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_78.2: init %i32.builtin = converted %int_3.loc4_49, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc4_78.3: ref %i32.builtin = array_index file.%arr.var, %int_0
// CHECK:STDOUT:   %.loc4_78.4: init %i32.builtin = initialize_from %.loc4_78.2 to %.loc4_78.3 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %impl.elem0.loc4_78.2: %.63c = impl_witness_access constants.%ImplicitAs.impl_witness.c15, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT: specific @As.Convert(constants.%T, constants.%Self.aa9) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.2c2
// CHECK:STDOUT:   %Self => constants.%Self.aa9
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.d87
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.8ac
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.e68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.a29
// CHECK:STDOUT:   %Self => constants.%Self.a6c
// CHECK:STDOUT:   %As.Convert.type => constants.%As.Convert.type.378
// CHECK:STDOUT:   %As.Convert => constants.%As.Convert.e51
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.bc2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.Op(constants.%T, constants.%Self.895) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.51e
// CHECK:STDOUT:   %Self => constants.%Self.895
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.b1f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.086
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.2a3
// CHECK:STDOUT:   %Self => constants.%Self.558
// CHECK:STDOUT:   %AddWith.Op.type => constants.%AddWith.Op.type.0b7
// CHECK:STDOUT:   %AddWith.Op => constants.%AddWith.Op.9d6
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.dff
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(constants.%T, constants.%Self.4f1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2d9
// CHECK:STDOUT:   %Self => constants.%Self.4f1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.ab0
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.c40
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.e68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7a9
// CHECK:STDOUT:   %Self => constants.%Self.886
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.71e
// CHECK:STDOUT:   %ImplicitAs.Convert => constants.%ImplicitAs.Convert.0e2
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.959
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5cf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.873
// CHECK:STDOUT:   %Self => constants.%Self.418
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.059
// CHECK:STDOUT:   %ImplicitAs.Convert => constants.%ImplicitAs.Convert.4d7
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.398
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5eb
// CHECK:STDOUT: }
// CHECK:STDOUT:

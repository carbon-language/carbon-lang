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
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.b35: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.b35 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.bc7: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self.deb: %AddWith.type.bc7 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.6cc: type = facet_access_type %Self.deb [symbolic]
// CHECK:STDOUT:   %pattern_type.4a8: type = pattern_type %Self.as_type.6cc [symbolic]
// CHECK:STDOUT:   %AddWith.Op.type.5b7: type = fn_type @AddWith.Op, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %AddWith.Op.037: %AddWith.Op.type.5b7 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.36e: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.adf: %AddWith.assoc_type.36e = assoc_entity element0, @AddWith.%AddWith.Op.decl [symbolic]
// CHECK:STDOUT:   %As.type.b51: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.b51 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.8ba: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.b4e: %As.type.8ba = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.7f0: type = facet_access_type %Self.b4e [symbolic]
// CHECK:STDOUT:   %pattern_type.947: type = pattern_type %Self.as_type.7f0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %As.Convert.type.ad1: type = fn_type @As.Convert, @As(%T) [symbolic]
// CHECK:STDOUT:   %As.Convert.0ed: %As.Convert.type.ad1 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type.a21: type = assoc_entity_type @As, @As(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.1d5: %As.assoc_type.a21 = assoc_entity element0, @As.%As.Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.07f: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.0f3: %ImplicitAs.type.07f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.419: type = facet_access_type %Self.0f3 [symbolic]
// CHECK:STDOUT:   %pattern_type.a93: type = pattern_type %Self.as_type.419 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4cf: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.147: %ImplicitAs.Convert.type.4cf = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.095: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.8f8: %ImplicitAs.assoc_type.095 = assoc_entity element0, @ImplicitAs.%ImplicitAs.Convert.decl [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %AddWith.type.49d: type = facet_type <@AddWith, @AddWith(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.da9: %AddWith.type.49d = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.Op.type.efc: type = fn_type @AddWith.Op, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %AddWith.Op.a62: %AddWith.Op.type.efc = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.a65: type = assoc_entity_type @AddWith, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.ee2: %AddWith.assoc_type.a65 = assoc_entity element0, @AddWith.%AddWith.Op.decl [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness file.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.956: type = pattern_type %i32.builtin [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.type: type = fn_type @i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op: %i32.builtin.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.49d = facet_value %i32.builtin, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %As.type.a09: type = facet_type <@As, @As(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.2fa: %As.type.a09 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.Convert.type.c0d: type = fn_type @As.Convert, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %As.Convert.713: %As.Convert.type.c0d = struct_value () [concrete]
// CHECK:STDOUT:   %As.assoc_type.b9b: type = assoc_entity_type @As, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.b02: %As.assoc_type.b9b = assoc_entity element0, @As.%As.Convert.decl [concrete]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness file.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a09 = facet_value Core.IntLiteral, (%As.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.11a: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.e0a: %ImplicitAs.type.11a = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.752: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.fcc: %ImplicitAs.Convert.type.752 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.1cf: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.3ce: %ImplicitAs.assoc_type.1cf = assoc_entity element0, @ImplicitAs.%ImplicitAs.Convert.decl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.be0: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc27 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.bcb: %ImplicitAs.type.11a = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.be0) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9fc: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Self.37e: %ImplicitAs.type.9fc = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.60e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.c73: %ImplicitAs.Convert.type.60e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.014: type = assoc_entity_type @ImplicitAs, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc0.757: %ImplicitAs.assoc_type.014 = assoc_entity element0, @ImplicitAs.%ImplicitAs.Convert.decl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.95b: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc31 [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert: %i32.builtin.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a85: %ImplicitAs.type.9fc = facet_value %i32.builtin, (%ImplicitAs.impl_witness.95b) [concrete]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = binding_pattern N [concrete]
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
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddWith.decl: %AddWith.type.b35 = interface_decl @AddWith [concrete = constants.%AddWith.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc7_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_19.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.b51 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc11_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc15_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.1 (constants.%T)]
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
// CHECK:STDOUT:     %AddWith.type: type = facet_type <@AddWith, @AddWith(constants.%i32.builtin)> [concrete = constants.%AddWith.type.49d]
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
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%i32.builtin)> [concrete = constants.%As.type.a09]
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
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32.builtin)> [concrete = constants.%ImplicitAs.type.11a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc27 = impl_witness_table (@Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.decl), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc27: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc27 [concrete = constants.%ImplicitAs.impl_witness.be0]
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
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete = constants.%ImplicitAs.type.9fc]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc31 = impl_witness_table (@i32.builtin.as.ImplicitAs.impl.%i32.builtin.as.ImplicitAs.impl.Convert.decl), @i32.builtin.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc31: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc31 [concrete = constants.%ImplicitAs.impl_witness.95b]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(%T.loc7_19.2: type) {
// CHECK:STDOUT:   %T.loc7_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_19.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T.loc7_19.1)> [symbolic = %AddWith.type (constants.%AddWith.type.bc7)]
// CHECK:STDOUT:   %Self.2: @AddWith.%AddWith.type (%AddWith.type.bc7) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.deb)]
// CHECK:STDOUT:   %AddWith.Op.type: type = fn_type @AddWith.Op, @AddWith(%T.loc7_19.1) [symbolic = %AddWith.Op.type (constants.%AddWith.Op.type.5b7)]
// CHECK:STDOUT:   %AddWith.Op: @AddWith.%AddWith.Op.type (%AddWith.Op.type.5b7) = struct_value () [symbolic = %AddWith.Op (constants.%AddWith.Op.037)]
// CHECK:STDOUT:   %AddWith.assoc_type: type = assoc_entity_type @AddWith, @AddWith(%T.loc7_19.1) [symbolic = %AddWith.assoc_type (constants.%AddWith.assoc_type.36e)]
// CHECK:STDOUT:   %assoc0.loc8_41.2: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.36e) = assoc_entity element0, %AddWith.Op.decl [symbolic = %assoc0.loc8_41.2 (constants.%assoc0.adf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @AddWith.%AddWith.type (%AddWith.type.bc7) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.deb)]
// CHECK:STDOUT:     %AddWith.Op.decl: @AddWith.%AddWith.Op.type (%AddWith.Op.type.5b7) = fn_decl @AddWith.Op [symbolic = @AddWith.%AddWith.Op (constants.%AddWith.Op.037)] {
// CHECK:STDOUT:       %self.patt: @AddWith.Op.%pattern_type (%pattern_type.4a8) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @AddWith.Op.%pattern_type (%pattern_type.4a8) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %other.patt: @AddWith.Op.%pattern_type (%pattern_type.4a8) = binding_pattern other [concrete]
// CHECK:STDOUT:       %other.param_patt: @AddWith.Op.%pattern_type (%pattern_type.4a8) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:       %return.patt: @AddWith.Op.%pattern_type (%pattern_type.4a8) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @AddWith.Op.%pattern_type (%pattern_type.4a8) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc8_37.1: @AddWith.Op.%AddWith.type (%AddWith.type.bc7) = specific_constant @AddWith.%Self.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.deb)]
// CHECK:STDOUT:       %Self.ref.loc8_37: @AddWith.Op.%AddWith.type (%AddWith.type.bc7) = name_ref Self, %.loc8_37.1 [symbolic = %Self (constants.%Self.deb)]
// CHECK:STDOUT:       %Self.as_type.loc8_37: type = facet_access_type %Self.ref.loc8_37 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.6cc)]
// CHECK:STDOUT:       %.loc8_37.2: type = converted %Self.ref.loc8_37, %Self.as_type.loc8_37 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.6cc)]
// CHECK:STDOUT:       %self.param: @AddWith.Op.%Self.as_type.loc8_15.1 (%Self.as_type.6cc) = value_param call_param0
// CHECK:STDOUT:       %.loc8_15.1: type = splice_block %.loc8_15.3 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.6cc)] {
// CHECK:STDOUT:         %.loc8_15.2: @AddWith.Op.%AddWith.type (%AddWith.type.bc7) = specific_constant @AddWith.%Self.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.deb)]
// CHECK:STDOUT:         %Self.ref.loc8_15: @AddWith.Op.%AddWith.type (%AddWith.type.bc7) = name_ref Self, %.loc8_15.2 [symbolic = %Self (constants.%Self.deb)]
// CHECK:STDOUT:         %Self.as_type.loc8_15.2: type = facet_access_type %Self.ref.loc8_15 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.6cc)]
// CHECK:STDOUT:         %.loc8_15.3: type = converted %Self.ref.loc8_15, %Self.as_type.loc8_15.2 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.6cc)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @AddWith.Op.%Self.as_type.loc8_15.1 (%Self.as_type.6cc) = bind_name self, %self.param
// CHECK:STDOUT:       %other.param: @AddWith.Op.%Self.as_type.loc8_15.1 (%Self.as_type.6cc) = value_param call_param1
// CHECK:STDOUT:       %.loc8_28.1: type = splice_block %.loc8_28.3 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.6cc)] {
// CHECK:STDOUT:         %.loc8_28.2: @AddWith.Op.%AddWith.type (%AddWith.type.bc7) = specific_constant @AddWith.%Self.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.deb)]
// CHECK:STDOUT:         %Self.ref.loc8_28: @AddWith.Op.%AddWith.type (%AddWith.type.bc7) = name_ref Self, %.loc8_28.2 [symbolic = %Self (constants.%Self.deb)]
// CHECK:STDOUT:         %Self.as_type.loc8_28: type = facet_access_type %Self.ref.loc8_28 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.6cc)]
// CHECK:STDOUT:         %.loc8_28.3: type = converted %Self.ref.loc8_28, %Self.as_type.loc8_28 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.6cc)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %other: @AddWith.Op.%Self.as_type.loc8_15.1 (%Self.as_type.6cc) = bind_name other, %other.param
// CHECK:STDOUT:       %return.param: ref @AddWith.Op.%Self.as_type.loc8_15.1 (%Self.as_type.6cc) = out_param call_param2
// CHECK:STDOUT:       %return: ref @AddWith.Op.%Self.as_type.loc8_15.1 (%Self.as_type.6cc) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc8_41.1: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.36e) = assoc_entity element0, %AddWith.Op.decl [symbolic = %assoc0.loc8_41.2 (constants.%assoc0.adf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Op = %assoc0.loc8_41.1
// CHECK:STDOUT:     witness = (%AddWith.Op.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%T.loc11_14.2: type) {
// CHECK:STDOUT:   %T.loc11_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T.loc11_14.1)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self.2: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:   %As.Convert.type: type = fn_type @As.Convert, @As(%T.loc11_14.1) [symbolic = %As.Convert.type (constants.%As.Convert.type.ad1)]
// CHECK:STDOUT:   %As.Convert: @As.%As.Convert.type (%As.Convert.type.ad1) = struct_value () [symbolic = %As.Convert (constants.%As.Convert.0ed)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%T.loc11_14.1) [symbolic = %As.assoc_type (constants.%As.assoc_type.a21)]
// CHECK:STDOUT:   %assoc0.loc12_32.2: @As.%As.assoc_type (%As.assoc_type.a21) = assoc_entity element0, %As.Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.1d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.b4e)]
// CHECK:STDOUT:     %As.Convert.decl: @As.%As.Convert.type (%As.Convert.type.ad1) = fn_decl @As.Convert [symbolic = @As.%As.Convert (constants.%As.Convert.0ed)] {
// CHECK:STDOUT:       %self.patt: @As.Convert.%pattern_type.loc12_14 (%pattern_type.947) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @As.Convert.%pattern_type.loc12_14 (%pattern_type.947) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @As.Convert.%pattern_type.loc12_28 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @As.Convert.%pattern_type.loc12_28 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @As.%T.loc11_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @As.Convert.%Self.as_type.loc12_20.1 (%Self.as_type.7f0) = value_param call_param0
// CHECK:STDOUT:       %.loc12_20.1: type = splice_block %.loc12_20.3 [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.7f0)] {
// CHECK:STDOUT:         %.loc12_20.2: @As.Convert.%As.type (%As.type.8ba) = specific_constant @As.%Self.1, @As(constants.%T) [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.ref: @As.Convert.%As.type (%As.type.8ba) = name_ref Self, %.loc12_20.2 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:         %Self.as_type.loc12_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:         %.loc12_20.3: type = converted %Self.ref, %Self.as_type.loc12_20.2 [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @As.Convert.%Self.as_type.loc12_20.1 (%Self.as_type.7f0) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @As.Convert.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @As.Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc12_32.1: @As.%As.assoc_type (%As.assoc_type.a21) = assoc_entity element0, %As.Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.1d5)]
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
// CHECK:STDOUT:   %T.loc15_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc15_22.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%T.loc15_22.1) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type.4cf)]
// CHECK:STDOUT:   %ImplicitAs.Convert: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.4cf) = struct_value () [symbolic = %ImplicitAs.Convert (constants.%ImplicitAs.Convert.147)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T.loc15_22.1) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.095)]
// CHECK:STDOUT:   %assoc0.loc16_32.2: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.095) = assoc_entity element0, %ImplicitAs.Convert.decl [symbolic = %assoc0.loc16_32.2 (constants.%assoc0.8f8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.0f3)]
// CHECK:STDOUT:     %ImplicitAs.Convert.decl: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.4cf) = fn_decl @ImplicitAs.Convert [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.147)] {
// CHECK:STDOUT:       %self.patt: @ImplicitAs.Convert.%pattern_type.loc16_14 (%pattern_type.a93) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @ImplicitAs.Convert.%pattern_type.loc16_14 (%pattern_type.a93) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @ImplicitAs.Convert.%pattern_type.loc16_28 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @ImplicitAs.Convert.%pattern_type.loc16_28 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc15_22.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.Convert.%Self.as_type.loc16_20.1 (%Self.as_type.419) = value_param call_param0
// CHECK:STDOUT:       %.loc16_20.1: type = splice_block %.loc16_20.3 [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.419)] {
// CHECK:STDOUT:         %.loc16_20.2: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = name_ref Self, %.loc16_20.2 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:         %Self.as_type.loc16_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:         %.loc16_20.3: type = converted %Self.ref, %Self.as_type.loc16_20.2 [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.Convert.%Self.as_type.loc16_20.1 (%Self.as_type.419) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @ImplicitAs.Convert.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ImplicitAs.Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc16_32.1: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.095) = assoc_entity element0, %ImplicitAs.Convert.decl [symbolic = %assoc0.loc16_32.2 (constants.%assoc0.8f8)]
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
// CHECK:STDOUT:     %self.patt: %pattern_type.956 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.956 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.956 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.956 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc20_37: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc19_6.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc20_15: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc19_6.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %i32.builtin = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc20_28: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc19_6.2 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %other: %i32.builtin = bind_name other, %other.param
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
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = binding_pattern self [concrete]
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
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
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
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = binding_pattern self [concrete]
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
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
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
// CHECK:STDOUT:     %self.patt: %pattern_type.956 = binding_pattern self [concrete]
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
// CHECK:STDOUT:     %self: %i32.builtin = bind_name self, %self.param
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
// CHECK:STDOUT: generic fn @AddWith.Op(@AddWith.%T.loc7_19.2: type, @AddWith.%Self.1: @AddWith.%AddWith.type (%AddWith.type.bc7)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.bc7)]
// CHECK:STDOUT:   %Self: @AddWith.Op.%AddWith.type (%AddWith.type.bc7) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.deb)]
// CHECK:STDOUT:   %Self.as_type.loc8_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.6cc)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc8_15.1 [symbolic = %pattern_type (constants.%pattern_type.4a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @AddWith.Op.%Self.as_type.loc8_15.1 (%Self.as_type.6cc), %other.param: @AddWith.Op.%Self.as_type.loc8_15.1 (%Self.as_type.6cc)) -> @AddWith.Op.%Self.as_type.loc8_15.1 (%Self.as_type.6cc);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @As.Convert(@As.%T.loc11_14.2: type, @As.%Self.1: @As.%As.type (%As.type.8ba)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.8ba)]
// CHECK:STDOUT:   %Self: @As.Convert.%As.type (%As.type.8ba) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.b4e)]
// CHECK:STDOUT:   %Self.as_type.loc12_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.7f0)]
// CHECK:STDOUT:   %pattern_type.loc12_14: type = pattern_type %Self.as_type.loc12_20.1 [symbolic = %pattern_type.loc12_14 (constants.%pattern_type.947)]
// CHECK:STDOUT:   %pattern_type.loc12_28: type = pattern_type %T [symbolic = %pattern_type.loc12_28 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @As.Convert.%Self.as_type.loc12_20.1 (%Self.as_type.7f0)) -> @As.Convert.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.Convert(@ImplicitAs.%T.loc15_22.2: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.07f)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.07f)]
// CHECK:STDOUT:   %Self: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.07f) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.0f3)]
// CHECK:STDOUT:   %Self.as_type.loc16_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.419)]
// CHECK:STDOUT:   %pattern_type.loc16_14: type = pattern_type %Self.as_type.loc16_20.1 [symbolic = %pattern_type.loc16_14 (constants.%pattern_type.a93)]
// CHECK:STDOUT:   %pattern_type.loc16_28: type = pattern_type %T [symbolic = %pattern_type.loc16_28 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ImplicitAs.Convert.%Self.as_type.loc16_20.1 (%Self.as_type.419)) -> @ImplicitAs.Convert.%T (%T);
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
// CHECK:STDOUT: specific @AddWith.Op(constants.%T, constants.%Self.deb) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.bc7
// CHECK:STDOUT:   %Self => constants.%Self.deb
// CHECK:STDOUT:   %Self.as_type.loc8_15.1 => constants.%Self.as_type.6cc
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4a8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.Convert(constants.%T, constants.%Self.b4e) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.8ba
// CHECK:STDOUT:   %Self => constants.%Self.b4e
// CHECK:STDOUT:   %Self.as_type.loc12_20.1 => constants.%Self.as_type.7f0
// CHECK:STDOUT:   %pattern_type.loc12_14 => constants.%pattern_type.947
// CHECK:STDOUT:   %pattern_type.loc12_28 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(constants.%T, constants.%Self.0f3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.07f
// CHECK:STDOUT:   %Self => constants.%Self.0f3
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => constants.%Self.as_type.419
// CHECK:STDOUT:   %pattern_type.loc16_14 => constants.%pattern_type.a93
// CHECK:STDOUT:   %pattern_type.loc16_28 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc7_19.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.49d
// CHECK:STDOUT:   %Self.2 => constants.%Self.da9
// CHECK:STDOUT:   %AddWith.Op.type => constants.%AddWith.Op.type.efc
// CHECK:STDOUT:   %AddWith.Op => constants.%AddWith.Op.a62
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.a65
// CHECK:STDOUT:   %assoc0.loc8_41.2 => constants.%assoc0.ee2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.Op(constants.%i32.builtin, constants.%AddWith.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.49d
// CHECK:STDOUT:   %Self => constants.%AddWith.facet
// CHECK:STDOUT:   %Self.as_type.loc8_15.1 => constants.%i32.builtin
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.956
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc11_14.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.a09
// CHECK:STDOUT:   %Self.2 => constants.%Self.2fa
// CHECK:STDOUT:   %As.Convert.type => constants.%As.Convert.type.c0d
// CHECK:STDOUT:   %As.Convert => constants.%As.Convert.713
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.b9b
// CHECK:STDOUT:   %assoc0.loc12_32.2 => constants.%assoc0.b02
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.Convert(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.a09
// CHECK:STDOUT:   %Self => constants.%As.facet
// CHECK:STDOUT:   %Self.as_type.loc12_20.1 => Core.IntLiteral
// CHECK:STDOUT:   %pattern_type.loc12_14 => constants.%pattern_type.dc0
// CHECK:STDOUT:   %pattern_type.loc12_28 => constants.%pattern_type.956
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.11a
// CHECK:STDOUT:   %Self.2 => constants.%Self.e0a
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.752
// CHECK:STDOUT:   %ImplicitAs.Convert => constants.%ImplicitAs.Convert.fcc
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.1cf
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.3ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(constants.%i32.builtin, constants.%ImplicitAs.facet.bcb) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.11a
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.bcb
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => Core.IntLiteral
// CHECK:STDOUT:   %pattern_type.loc16_14 => constants.%pattern_type.dc0
// CHECK:STDOUT:   %pattern_type.loc16_28 => constants.%pattern_type.956
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc15_22.1 => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9fc
// CHECK:STDOUT:   %Self.2 => constants.%Self.37e
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.60e
// CHECK:STDOUT:   %ImplicitAs.Convert => constants.%ImplicitAs.Convert.c73
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.014
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.757
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(Core.IntLiteral, constants.%ImplicitAs.facet.a85) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9fc
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.a85
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => constants.%i32.builtin
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %As.type.eed: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.65a: %As.type.eed = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.Convert.type.843: type = fn_type @As.Convert, @As(%T) [symbolic]
// CHECK:STDOUT:   %As.Convert.95f: %As.Convert.type.843 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Self.as_type.04d: type = facet_access_type %Self.65a [symbolic]
// CHECK:STDOUT:   %pattern_type.ca4: type = pattern_type %Self.as_type.04d [symbolic]
// CHECK:STDOUT:   %As.assoc_type.760: type = assoc_entity_type @As, @As(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.076: %As.assoc_type.760 = assoc_entity element0, imports.%Core.import_ref.708 [symbolic]
// CHECK:STDOUT:   %As.type.a6d: type = facet_type <@As, @As(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.c25: %As.type.a6d = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %As.Convert.type.378: type = fn_type @As.Convert, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %As.Convert.e51: %As.Convert.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %As.assoc_type.bc2: type = assoc_entity_type @As, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.9fb: %As.assoc_type.bc2 = assoc_entity element0, imports.%Core.import_ref.708 [concrete]
// CHECK:STDOUT:   %assoc0.97d: %As.assoc_type.760 = assoc_entity element0, imports.%Core.import_ref.4e8 [symbolic]
// CHECK:STDOUT:   %AddWith.type.e05: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.e05 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.6d3: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self.6c1: %AddWith.type.6d3 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.type.6f5: type = facet_type <@AddWith, @AddWith(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %AddWith.Op.type.22d: type = fn_type @AddWith.Op, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %AddWith.Op.965: %AddWith.Op.type.22d = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.030: type = facet_access_type %Self.6c1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7fc: type = pattern_type %Self.as_type.030 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.c10: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.89962d.1: %AddWith.assoc_type.c10 = assoc_entity element0, imports.%Core.import_ref.7e6ace.1 [symbolic]
// CHECK:STDOUT:   %Self.ec3: %AddWith.type.6f5 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.Op.type.0b7: type = fn_type @AddWith.Op, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %AddWith.Op.9d6: %AddWith.Op.type.0b7 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.dff: type = assoc_entity_type @AddWith, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.7f1: %AddWith.assoc_type.dff = assoc_entity element0, imports.%Core.import_ref.1b9 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d62: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.519: %ImplicitAs.type.d62 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.61e: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.275: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.42e: %ImplicitAs.Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.40a: type = facet_access_type %Self.519 [symbolic]
// CHECK:STDOUT:   %pattern_type.f3e: type = pattern_type %Self.as_type.40a [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.dc001e.1: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic]
// CHECK:STDOUT:   %Self.d28: %ImplicitAs.type.61e = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.059: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.4d7: %ImplicitAs.Convert.type.059 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.398: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.6fd: %ImplicitAs.assoc_type.398 = assoc_entity element0, imports.%Core.import_ref.1c752f.1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Self.9ac: %ImplicitAs.type.2fd = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.0e2: %ImplicitAs.Convert.type.71e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.959: type = assoc_entity_type @ImplicitAs, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc0.8c4: %ImplicitAs.assoc_type.959 = assoc_entity element0, imports.%Core.import_ref.1c752f.2 [concrete]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness imports.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a6d = facet_value Core.IntLiteral, (%As.impl_witness) [concrete]
// CHECK:STDOUT:   %.387: type = fn_type_with_self_type %As.Convert.type.378, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.b34: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.f38: %i32.builtin = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.324: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_2.5a1: %i32.builtin = int_value 2 [concrete]
// CHECK:STDOUT:   %assoc0.89962d.2: %AddWith.assoc_type.c10 = assoc_entity element0, imports.%Core.import_ref.7e6ace.2 [symbolic]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness imports.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.6f5 = facet_value %i32.builtin, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.1e7: type = fn_type_with_self_type %AddWith.Op.type.0b7, %AddWith.facet [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.type: type = fn_type @i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op: %i32.builtin.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.bound.de8: <bound method> = bound_method %int_1.f38, %i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_3.a0f: %i32.builtin = int_value 3 [concrete]
// CHECK:STDOUT:   %assoc0.dc001e.2: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207961.2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9c5: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5fc [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.07b: %ImplicitAs.type.2fd = facet_value %i32.builtin, (%ImplicitAs.impl_witness.9c5) [concrete]
// CHECK:STDOUT:   %.53b: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.07b [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert: %i32.builtin.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.a0f, %i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.9e2: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.94d: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.8fc: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_4.4f1: %i32.builtin = int_value 4 [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.bound.703: <bound method> = bound_method %int_3.a0f, %i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_7: %i32.builtin = int_value 7 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, %i32.builtin) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.07a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.78e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a3d: %ImplicitAs.type.61e = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.07a) [concrete]
// CHECK:STDOUT:   %.ec1: type = fn_type_with_self_type %ImplicitAs.Convert.type.059, %ImplicitAs.facet.a3d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b6b: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.626: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.1: type = import_ref Core//default, loc11_14, loaded [symbolic = @As.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.a7c = import_ref Core//default, inst106 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.5e1: @As.%As.assoc_type (%As.assoc_type.760) = import_ref Core//default, loc12_32, loaded [symbolic = @As.%assoc0 (constants.%assoc0.97d)]
// CHECK:STDOUT:   %Core.Convert.313 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.2: type = import_ref Core//default, loc11_14, loaded [symbolic = @As.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.996: @As.%As.type (%As.type.eed) = import_ref Core//default, inst106 [no loc], loaded [symbolic = @As.%Self (constants.%Self.65a)]
// CHECK:STDOUT:   %Core.import_ref.708: @As.%As.Convert.type (%As.Convert.type.843) = import_ref Core//default, loc12_32, loaded [symbolic = @As.%As.Convert (constants.%As.Convert.95f)]
// CHECK:STDOUT:   %Core.import_ref.4e8 = import_ref Core//default, loc12_32, unloaded
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.3: type = import_ref Core//default, loc7_19, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.476 = import_ref Core//default, inst51 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.35d: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.c10) = import_ref Core//default, loc8_41, loaded [symbolic = @AddWith.%assoc0 (constants.%assoc0.89962d.2)]
// CHECK:STDOUT:   %Core.Op = import_ref Core//default, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.772: <witness> = import_ref Core//default, loc19_26, loaded [concrete = constants.%AddWith.impl_witness]
// CHECK:STDOUT:   %Core.import_ref.c8c7cd.1: type = import_ref Core//default, loc19_6, loaded [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %Core.import_ref.ef3: type = import_ref Core//default, loc19_24, loaded [concrete = constants.%AddWith.type.6f5]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.4: type = import_ref Core//default, loc7_19, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.6bf: @AddWith.%AddWith.type (%AddWith.type.6d3) = import_ref Core//default, inst51 [no loc], loaded [symbolic = @AddWith.%Self (constants.%Self.6c1)]
// CHECK:STDOUT:   %Core.import_ref.7e6ace.1 = import_ref Core//default, loc8_41, unloaded
// CHECK:STDOUT:   %Core.import_ref.1b9: @AddWith.%AddWith.Op.type (%AddWith.Op.type.22d) = import_ref Core//default, loc8_41, loaded [symbolic = @AddWith.%AddWith.Op (constants.%AddWith.Op.965)]
// CHECK:STDOUT:   %Core.import_ref.cb6: <witness> = import_ref Core//default, loc23_30, loaded [concrete = constants.%As.impl_witness]
// CHECK:STDOUT:   %Core.import_ref.8721d7.1: type = import_ref Core//default, loc23_17, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.1e5: type = import_ref Core//default, loc23_28, loaded [concrete = constants.%As.type.a6d]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.5: type = import_ref Core//default, loc15_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ff5 = import_ref Core//default, inst151 [no loc], unloaded
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//default, loc16_32, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc001e.2)]
// CHECK:STDOUT:   %Core.Convert.e69 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.c62: <witness> = import_ref Core//default, loc27_38, loaded [concrete = constants.%ImplicitAs.impl_witness.07a]
// CHECK:STDOUT:   %Core.import_ref.8721d7.2: type = import_ref Core//default, loc27_17, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.4d9: type = import_ref Core//default, loc27_36, loaded [concrete = constants.%ImplicitAs.type.61e]
// CHECK:STDOUT:   %Core.import_ref.5ab3ec.6: type = import_ref Core//default, loc15_22, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = import_ref Core//default, inst151 [no loc], loaded [symbolic = @ImplicitAs.%Self (constants.%Self.519)]
// CHECK:STDOUT:   %Core.import_ref.207961.1 = import_ref Core//default, loc16_32, unloaded
// CHECK:STDOUT:   %Core.import_ref.1c752f.1: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.275) = import_ref Core//default, loc16_32, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.c5f: <witness> = import_ref Core//default, loc31_38, loaded [concrete = constants.%ImplicitAs.impl_witness.9c5]
// CHECK:STDOUT:   %Core.import_ref.c8c7cd.2: type = import_ref Core//default, loc31_6, loaded [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %Core.import_ref.efb: type = import_ref Core//default, loc31_36, loaded [concrete = constants.%ImplicitAs.type.2fd]
// CHECK:STDOUT:   %Core.import_ref.1c752f.2: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.275) = import_ref Core//default, loc16_32, loaded [symbolic = @ImplicitAs.%ImplicitAs.Convert (constants.%ImplicitAs.Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.73a: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//default, loc24_35, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.73a), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.AddWith: %AddWith.type.e05 = import_ref Core//default, AddWith, loaded [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:   %Core.import_ref.7e6ace.2 = import_ref Core//default, loc8_41, unloaded
// CHECK:STDOUT:   %Core.import_ref.079: %i32.builtin.as.AddWith.impl.Op.type = import_ref Core//default, loc20_42, loaded [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:   %AddWith.impl_witness_table = impl_witness_table (%Core.import_ref.079), @i32.builtin.as.AddWith.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.207961.2 = import_ref Core//default, loc16_32, unloaded
// CHECK:STDOUT:   %Core.import_ref.4f9: %i32.builtin.as.ImplicitAs.impl.Convert.type = import_ref Core//default, loc32_44, loaded [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5fc = impl_witness_table (%Core.import_ref.4f9), @i32.builtin.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f35: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//default, loc28_35, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.78e = impl_witness_table (%Core.import_ref.f35), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %arr.patt: %pattern_type.9e2 = binding_pattern arr [concrete]
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
// CHECK:STDOUT:     %impl.elem0.loc4_24: %.387 = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_24: <bound method> = bound_method %int_1, %impl.elem0.loc4_24 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.b34]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call.loc4_24: init %i32.builtin = call %bound_method.loc4_24(%int_1) [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_24.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_24 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_24.2: %i32.builtin = converted %int_1, %.loc4_24.1 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %int_32.loc4_40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %Int.call.loc4_40: init type = call constants.%Int(%int_32.loc4_40) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_40.1: type = value_of_initializer %Int.call.loc4_40 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_40.2: type = converted %Int.call.loc4_40, %.loc4_40.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_37: %.387 = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_37: <bound method> = bound_method %int_2, %impl.elem0.loc4_37 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.324]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call.loc4_37: init %i32.builtin = call %bound_method.loc4_37(%int_2) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_37.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_37 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_37.2: %i32.builtin = converted %int_2, %.loc4_37.1 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %impl.elem0.loc4_32.1: %.1e7 = impl_witness_access constants.%AddWith.impl_witness, element0 [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:     %bound_method.loc4_32.1: <bound method> = bound_method %.loc4_24.2, %impl.elem0.loc4_32.1 [concrete = constants.%i32.builtin.as.AddWith.impl.Op.bound.de8]
// CHECK:STDOUT:     %i32.builtin.as.AddWith.impl.Op.call: init %i32.builtin = call %bound_method.loc4_32.1(%.loc4_24.2, %.loc4_37.2) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc4_16.1: type = value_of_initializer %Int.call.loc4_16 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc4_16.2: type = converted %Int.call.loc4_16, %.loc4_16.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_32.2: %.53b = impl_witness_access constants.%ImplicitAs.impl_witness.9c5, element0 [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_32.2: <bound method> = bound_method %i32.builtin.as.AddWith.impl.Op.call, %impl.elem0.loc4_32.2 [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %.loc4_32.1: %i32.builtin = value_of_initializer %i32.builtin.as.AddWith.impl.Op.call [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc4_32.2: %i32.builtin = converted %i32.builtin.as.AddWith.impl.Op.call, %.loc4_32.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %i32.builtin.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc4_32.2(%.loc4_32.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc4_32.3: Core.IntLiteral = value_of_initializer %i32.builtin.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc4_32.4: Core.IntLiteral = converted %i32.builtin.as.AddWith.impl.Op.call, %.loc4_32.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %.loc4_32.4, %.loc4_16.2 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var [concrete = %arr.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(imports.%Core.import_ref.5ab3ec.1: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.eed)]
// CHECK:STDOUT:   %Self: @As.%As.type (%As.type.eed) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.65a)]
// CHECK:STDOUT:   %As.Convert.type: type = fn_type @As.Convert, @As(%T) [symbolic = %As.Convert.type (constants.%As.Convert.type.843)]
// CHECK:STDOUT:   %As.Convert: @As.%As.Convert.type (%As.Convert.type.843) = struct_value () [symbolic = %As.Convert (constants.%As.Convert.95f)]
// CHECK:STDOUT:   %As.assoc_type: type = assoc_entity_type @As, @As(%T) [symbolic = %As.assoc_type (constants.%As.assoc_type.760)]
// CHECK:STDOUT:   %assoc0: @As.%As.assoc_type (%As.assoc_type.760) = assoc_entity element0, imports.%Core.import_ref.708 [symbolic = %assoc0 (constants.%assoc0.076)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.a7c
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.5e1
// CHECK:STDOUT:     witness = (imports.%Core.Convert.313)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(imports.%Core.import_ref.5ab3ec.3: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.6d3)]
// CHECK:STDOUT:   %Self: @AddWith.%AddWith.type (%AddWith.type.6d3) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.6c1)]
// CHECK:STDOUT:   %AddWith.Op.type: type = fn_type @AddWith.Op, @AddWith(%T) [symbolic = %AddWith.Op.type (constants.%AddWith.Op.type.22d)]
// CHECK:STDOUT:   %AddWith.Op: @AddWith.%AddWith.Op.type (%AddWith.Op.type.22d) = struct_value () [symbolic = %AddWith.Op (constants.%AddWith.Op.965)]
// CHECK:STDOUT:   %AddWith.assoc_type: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic = %AddWith.assoc_type (constants.%AddWith.assoc_type.c10)]
// CHECK:STDOUT:   %assoc0: @AddWith.%AddWith.assoc_type (%AddWith.assoc_type.c10) = assoc_entity element0, imports.%Core.import_ref.7e6ace.1 [symbolic = %assoc0 (constants.%assoc0.89962d.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.476
// CHECK:STDOUT:     .Op = imports.%Core.import_ref.35d
// CHECK:STDOUT:     witness = (imports.%Core.Op)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.5ab3ec.5: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %ImplicitAs.Convert.type: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%T) [symbolic = %ImplicitAs.Convert.type (constants.%ImplicitAs.Convert.type.275)]
// CHECK:STDOUT:   %ImplicitAs.Convert: @ImplicitAs.%ImplicitAs.Convert.type (%ImplicitAs.Convert.type.275) = struct_value () [symbolic = %ImplicitAs.Convert (constants.%ImplicitAs.Convert.42e)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.ca0)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = assoc_entity element0, imports.%Core.import_ref.207961.1 [symbolic = %assoc0 (constants.%assoc0.dc001e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ff5
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.492
// CHECK:STDOUT:     witness = (imports.%Core.Convert.e69)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.AddWith.impl: imports.%Core.import_ref.c8c7cd.1 as imports.%Core.import_ref.ef3 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.772
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.As.impl: imports.%Core.import_ref.8721d7.1 as imports.%Core.import_ref.1e5 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.ImplicitAs.impl: imports.%Core.import_ref.8721d7.2 as imports.%Core.import_ref.4d9 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.c62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.ImplicitAs.impl: imports.%Core.import_ref.c8c7cd.2 as imports.%Core.import_ref.efb [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.c5f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @As.Convert(imports.%Core.import_ref.5ab3ec.2: type, imports.%Core.import_ref.996: @As.%As.type (%As.type.eed)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.eed)]
// CHECK:STDOUT:   %Self: @As.Convert.%As.type (%As.type.eed) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.65a)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.04d)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.ca4)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.Op(imports.%Core.import_ref.5ab3ec.4: type, imports.%Core.import_ref.6bf: @AddWith.%AddWith.type (%AddWith.type.6d3)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.6d3)]
// CHECK:STDOUT:   %Self: @AddWith.Op.%AddWith.type (%AddWith.type.6d3) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.6c1)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.030)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.7fc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.Convert(imports.%Core.import_ref.5ab3ec.6: type, imports.%Core.import_ref.ce1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d62)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d62)]
// CHECK:STDOUT:   %Self: @ImplicitAs.Convert.%ImplicitAs.type (%ImplicitAs.type.d62) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.519)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.40a)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.f3e)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Core.IntLiteral.as.As.impl.Convert = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.builtin.as.AddWith.impl.Op = "int.sadd" [from "core.carbon"];
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
// CHECK:STDOUT:   %impl.elem0.loc4_58: %.387 = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_58: <bound method> = bound_method %int_3.loc4_56, %impl.elem0.loc4_58 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.94d]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc4_58: init %i32.builtin = call %bound_method.loc4_58(%int_3.loc4_56) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_58.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_58 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_58.2: %i32.builtin = converted %int_3.loc4_56, %.loc4_58.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %int_4.loc4_69: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %int_32.loc4_74: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %Int.call.loc4_74: init type = call constants.%Int(%int_32.loc4_74) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_74.1: type = value_of_initializer %Int.call.loc4_74 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_74.2: type = converted %Int.call.loc4_74, %.loc4_74.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_71: %.387 = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_71: <bound method> = bound_method %int_4.loc4_69, %impl.elem0.loc4_71 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.8fc]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc4_71: init %i32.builtin = call %bound_method.loc4_71(%int_4.loc4_69) [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_71.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_71 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_71.2: %i32.builtin = converted %int_4.loc4_69, %.loc4_71.1 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %impl.elem0.loc4_66: %.1e7 = impl_witness_access constants.%AddWith.impl_witness, element0 [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc4_66: <bound method> = bound_method %.loc4_58.2, %impl.elem0.loc4_66 [concrete = constants.%i32.builtin.as.AddWith.impl.Op.bound.703]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.call: init %i32.builtin = call %bound_method.loc4_66(%.loc4_58.2, %.loc4_71.2) [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc4_78.1: %tuple.type = tuple_literal (%int_3.loc4_49, %int_4.loc4_52, %i32.builtin.as.AddWith.impl.Op.call)
// CHECK:STDOUT:   %impl.elem0.loc4_78.1: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_78.1: <bound method> = bound_method %int_3.loc4_49, %impl.elem0.loc4_78.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b6b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.1: init %i32.builtin = call %bound_method.loc4_78.1(%int_3.loc4_49) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_78.2: init %i32.builtin = converted %int_3.loc4_49, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc4_78.3: ref %i32.builtin = array_index file.%arr.var, %int_0
// CHECK:STDOUT:   %.loc4_78.4: init %i32.builtin = initialize_from %.loc4_78.2 to %.loc4_78.3 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %impl.elem0.loc4_78.2: %.ec1 = impl_witness_access constants.%ImplicitAs.impl_witness.07a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_78.2: <bound method> = bound_method %int_4.loc4_52, %impl.elem0.loc4_78.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.626]
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
// CHECK:STDOUT: specific @As.Convert(constants.%T, constants.%Self.65a) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.eed
// CHECK:STDOUT:   %Self => constants.%Self.65a
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.04d
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.ca4
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.a6d
// CHECK:STDOUT:   %Self => constants.%Self.c25
// CHECK:STDOUT:   %As.Convert.type => constants.%As.Convert.type.378
// CHECK:STDOUT:   %As.Convert => constants.%As.Convert.e51
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.bc2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9fb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.6f5
// CHECK:STDOUT:   %Self => constants.%Self.ec3
// CHECK:STDOUT:   %AddWith.Op.type => constants.%AddWith.Op.type.0b7
// CHECK:STDOUT:   %AddWith.Op => constants.%AddWith.Op.9d6
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.dff
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7f1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.Op(constants.%T, constants.%Self.6c1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.6d3
// CHECK:STDOUT:   %Self => constants.%Self.6c1
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.030
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7fc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.61e
// CHECK:STDOUT:   %Self => constants.%Self.d28
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.059
// CHECK:STDOUT:   %ImplicitAs.Convert => constants.%ImplicitAs.Convert.4d7
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.398
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6fd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.Convert(constants.%T, constants.%Self.519) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d62
// CHECK:STDOUT:   %Self => constants.%Self.519
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.40a
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.f3e
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2fd
// CHECK:STDOUT:   %Self => constants.%Self.9ac
// CHECK:STDOUT:   %ImplicitAs.Convert.type => constants.%ImplicitAs.Convert.type.71e
// CHECK:STDOUT:   %ImplicitAs.Convert => constants.%ImplicitAs.Convert.0e2
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.959
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8c4
// CHECK:STDOUT: }
// CHECK:STDOUT:

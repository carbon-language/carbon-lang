// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/no_prelude/call_from_operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/no_prelude/call_from_operator.carbon

// --- core.carbon

package Core;

fn IntLiteral() -> type = "int_literal.make_type";
fn Int(N: IntLiteral()) -> type = "int.make_type_signed";

interface Add {
  fn Op[self: Self](other: Self) -> Self;
}

interface As(T:! type) {
  fn Convert[self: Self]() -> T;
}

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

impl i32 as Add {
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

var arr: [i32; (1 as i32) + (2 as i32)] = (3, 4, (3 as i32) + (4 as i32));

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [template]
// CHECK:STDOUT:   %Self.1: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self.1 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %Add.type, %Op.type.1 [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @Add.%Op.decl [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %As.type.1: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As: %As.type.1 = struct_value () [template]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.2: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %.4: type = facet_access_type %Self.2 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @As(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %As.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.3: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %.7: type = facet_access_type %Self.3 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.8: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.2 [symbolic]
// CHECK:STDOUT:   %.9: %.8 = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %.10: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.10 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.11: %Add.type = facet_value %i32, %i32 [template]
// CHECK:STDOUT:   %.12: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.1, @As(%i32) [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %.13: type = assoc_entity_type %As.type.3, %Convert.type.4 [template]
// CHECK:STDOUT:   %.14: %.13 = assoc_entity element0, @As.%Convert.decl [template]
// CHECK:STDOUT:   %.15: %As.type.2 = facet_value Core.IntLiteral, Core.IntLiteral [symbolic]
// CHECK:STDOUT:   %.16: <witness> = interface_witness (%Convert.3) [template]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.4 [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.2, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.17: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.6 [template]
// CHECK:STDOUT:   %.18: %.17 = assoc_entity element0, @ImplicitAs.%Convert.decl [template]
// CHECK:STDOUT:   %.19: %ImplicitAs.type.2 = facet_value Core.IntLiteral, Core.IntLiteral [symbolic]
// CHECK:STDOUT:   %.20: <witness> = interface_witness (%Convert.5) [template]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.2, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [template]
// CHECK:STDOUT:   %.21: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.8 [template]
// CHECK:STDOUT:   %.22: %.21 = assoc_entity element0, @ImplicitAs.%Convert.decl [template]
// CHECK:STDOUT:   %.23: %ImplicitAs.type.2 = facet_value %i32, %i32 [symbolic]
// CHECK:STDOUT:   %.24: <witness> = interface_witness (%Convert.7) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [template = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [template = constants.%Int] {
// CHECK:STDOUT:     %N.patt: Core.IntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_22.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_22.2: type = converted %int_literal.make_type, %.loc5_22.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.decl: type = interface_decl @Add [template = constants.%Add.type] {} {}
// CHECK:STDOUT:   %As.decl: %As.type.1 = interface_decl @As [template = constants.%As] {
// CHECK:STDOUT:     %T.patt.loc11_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_14.1, runtime_param<invalid> [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.1 = interface_decl @ImplicitAs [template = constants.%ImplicitAs] {
// CHECK:STDOUT:     %T.patt.loc15_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc15_22.1, runtime_param<invalid> [symbolic = %T.patt.loc15_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc15_22.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc15_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %.loc19_6.1: Core.IntLiteral = int_value 32 [template = constants.%.10]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc19_6.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_6.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_6.3: type = converted %int.make_type_signed, %.loc19_6.2 [template = constants.%i32]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, file.%Add.decl [template = constants.%Add.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_17.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_17.2: type = converted %int_literal.make_type, %.loc23_17.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %As.ref: %As.type.1 = name_ref As, file.%As.decl [template = constants.%As]
// CHECK:STDOUT:     %.loc23_25: Core.IntLiteral = int_value 32 [template = constants.%.10]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc23_25) [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_24.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_24.2: type = converted %int.make_type_signed, %.loc23_24.1 [template = constants.%i32]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%i32)> [template = constants.%As.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc27_17.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc27_17.2: type = converted %int_literal.make_type, %.loc27_17.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, file.%ImplicitAs.decl [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %.loc27_33: Core.IntLiteral = int_value 32 [template = constants.%.10]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc27_33) [template = constants.%i32]
// CHECK:STDOUT:     %.loc27_32.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc27_32.2: type = converted %int.make_type_signed, %.loc27_32.1 [template = constants.%i32]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32)> [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %.loc31_6.1: Core.IntLiteral = int_value 32 [template = constants.%.10]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc31_6.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc31_6.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc31_6.3: type = converted %int.make_type_signed, %.loc31_6.2 [template = constants.%i32]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, file.%ImplicitAs.decl [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_23.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_23.2: type = converted %int_literal.make_type, %.loc31_23.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %Op.decl: %Op.type.1 = fn_decl @Op.1 [template = constants.%Op.1] {
// CHECK:STDOUT:     %self.patt: @Op.1.%.loc8_15.1 (%.1) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%.loc8_15.1 (%.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%.loc8_15.1 (%.1) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%.loc8_15.1 (%.1) = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%.loc8_15.1 (%.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%.loc8_15.1 (%.1) = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc8_15: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc8_15.2: type = facet_access_type %Self.ref.loc8_15 [symbolic = %.loc8_15.1 (constants.%.1)]
// CHECK:STDOUT:     %.loc8_15.3: type = converted %Self.ref.loc8_15, %.loc8_15.2 [symbolic = %.loc8_15.1 (constants.%.1)]
// CHECK:STDOUT:     %Self.ref.loc8_28: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc8_28.1: type = facet_access_type %Self.ref.loc8_28 [symbolic = %.loc8_15.1 (constants.%.1)]
// CHECK:STDOUT:     %.loc8_28.2: type = converted %Self.ref.loc8_28, %.loc8_28.1 [symbolic = %.loc8_15.1 (constants.%.1)]
// CHECK:STDOUT:     %Self.ref.loc8_37: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc8_37.1: type = facet_access_type %Self.ref.loc8_37 [symbolic = %.loc8_15.1 (constants.%.1)]
// CHECK:STDOUT:     %.loc8_37.2: type = converted %Self.ref.loc8_37, %.loc8_37.1 [symbolic = %.loc8_15.1 (constants.%.1)]
// CHECK:STDOUT:     %self.param: @Op.1.%.loc8_15.1 (%.1) = value_param runtime_param0
// CHECK:STDOUT:     %self: @Op.1.%.loc8_15.1 (%.1) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%.loc8_15.1 (%.1) = value_param runtime_param1
// CHECK:STDOUT:     %other: @Op.1.%.loc8_15.1 (%.1) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%.loc8_15.1 (%.1) = out_param runtime_param2
// CHECK:STDOUT:     %return: ref @Op.1.%.loc8_15.1 (%.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: %.2 = assoc_entity element0, %Op.decl [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc8
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%T.loc11_14.1: type) {
// CHECK:STDOUT:   %T.loc11_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T.loc11_14.2)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self.2: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%T.loc11_14.2) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.loc12_32.2: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.1) [symbolic = %.loc12_32.2 (constants.%.5)]
// CHECK:STDOUT:   %.loc12_32.3: @As.%.loc12_32.2 (%.5) = assoc_entity element0, %Convert.decl [symbolic = %.loc12_32.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.1) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.1)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%.loc12_20.1 (%.4) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%.loc12_20.1 (%.4) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc12_20.2: @Convert.1.%As.type (%As.type.2) = specific_constant @As.%Self.1, @As(constants.%T) [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:       %Self.ref: @Convert.1.%As.type (%As.type.2) = name_ref Self, %.loc12_20.2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:       %.loc12_20.3: type = facet_access_type %Self.ref [symbolic = %.loc12_20.1 (constants.%.4)]
// CHECK:STDOUT:       %.loc12_20.4: type = converted %Self.ref, %.loc12_20.3 [symbolic = %.loc12_20.1 (constants.%.4)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @As.%T.loc11_14.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.1.%.loc12_20.1 (%.4) = value_param runtime_param0
// CHECK:STDOUT:       %self: @Convert.1.%.loc12_20.1 (%.4) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc12_32.1: @As.%.loc12_32.2 (%.5) = assoc_entity element0, %Convert.decl [symbolic = %.loc12_32.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Convert = %.loc12_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc15_22.1: type) {
// CHECK:STDOUT:   %T.loc15_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc15_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc15_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%T.loc15_22.2) [symbolic = %Convert.type (constants.%Convert.type.2)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.2) = struct_value () [symbolic = %Convert (constants.%Convert.2)]
// CHECK:STDOUT:   %.loc16_32.2: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.2) [symbolic = %.loc16_32.2 (constants.%.8)]
// CHECK:STDOUT:   %.loc16_32.3: @ImplicitAs.%.loc16_32.2 (%.8) = assoc_entity element0, %Convert.decl [symbolic = %.loc16_32.3 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.3)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.2) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.2)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%.loc16_20.1 (%.7) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%.loc16_20.1 (%.7) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc16_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.2) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:       %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.2) = name_ref Self, %.loc16_20.2 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:       %.loc16_20.3: type = facet_access_type %Self.ref [symbolic = %.loc16_20.1 (constants.%.7)]
// CHECK:STDOUT:       %.loc16_20.4: type = converted %Self.ref, %.loc16_20.3 [symbolic = %.loc16_20.1 (constants.%.7)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc15_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.2.%.loc16_20.1 (%.7) = value_param runtime_param0
// CHECK:STDOUT:       %self: @Convert.2.%.loc16_20.1 (%.7) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc16_32.1: @ImplicitAs.%.loc16_32.2 (%.8) = assoc_entity element0, %Convert.decl [symbolic = %.loc16_32.3 (constants.%.9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Convert = %.loc16_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc19_6.3 as %Add.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.2 = fn_decl @Op.2 [template = constants.%Op.2] {
// CHECK:STDOUT:     %self.patt: %i32 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %i32 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %i32 = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %i32 = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc20_15: type = name_ref Self, @impl.1.%.loc19_6.3 [template = constants.%i32]
// CHECK:STDOUT:     %Self.ref.loc20_28: type = name_ref Self, @impl.1.%.loc19_6.3 [template = constants.%i32]
// CHECK:STDOUT:     %Self.ref.loc20_37: type = name_ref Self, @impl.1.%.loc19_6.3 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %self: %i32 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %other: %i32 = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_17: <witness> = interface_witness (%Op.decl) [template = constants.%.12]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.loc19_17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %.loc23_17.2 as %As.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.3 = fn_decl @Convert.3 [template = constants.%Convert.3] {
// CHECK:STDOUT:     %self.patt: Core.IntLiteral = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: Core.IntLiteral = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.2.%.loc23_17.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc24_31.1: Core.IntLiteral = int_value 32 [template = constants.%.10]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc24_31.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc24_31.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc24_31.3: type = converted %int.make_type_signed, %.loc24_31.2 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23_30: <witness> = interface_witness (%Convert.decl) [template = constants.%.16]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc23_30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %.loc27_17.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.5 = fn_decl @Convert.4 [template = constants.%Convert.5] {
// CHECK:STDOUT:     %self.patt: Core.IntLiteral = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: Core.IntLiteral = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.3.%.loc27_17.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc28_31.1: Core.IntLiteral = int_value 32 [template = constants.%.10]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc28_31.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc28_31.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc28_31.3: type = converted %int.make_type_signed, %.loc28_31.2 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc27_38: <witness> = interface_witness (%Convert.decl) [template = constants.%.20]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc27_38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %.loc31_6.3 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.7 = fn_decl @Convert.5 [template = constants.%Convert.7] {
// CHECK:STDOUT:     %self.patt: %i32 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %i32 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.4.%.loc31_6.3 [template = constants.%i32]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc32_42.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc32_42.2: type = converted %int_literal.make_type, %.loc32_42.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %self.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %self: %i32 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31_38: <witness> = interface_witness (%Convert.decl) [template = constants.%.24]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc31_38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@Add.%Self: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:   %.loc8_15.1: type = facet_access_type %Self [symbolic = %.loc8_15.1 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%.loc8_15.1 (%.1)](%other.param_patt: @Op.1.%.loc8_15.1 (%.1)) -> @Op.1.%.loc8_15.1 (%.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%T.loc11_14.1: type, @As.%Self.1: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %.loc12_20.1: type = facet_access_type %Self [symbolic = %.loc12_20.1 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%.loc12_20.1 (%.4)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%T.loc15_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %.loc16_20.1: type = facet_access_type %Self [symbolic = %.loc16_20.1 (constants.%.7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%.loc16_20.1 (%.7)]() -> @Convert.2.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self.param_patt: %i32](%other.param_patt: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: Core.IntLiteral]() -> %i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: Core.IntLiteral]() -> %i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: %i32]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT:   %.loc8_15.1 => constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%T) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.2) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %.loc12_20.1 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%T.loc11_14.2) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc15_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc15_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%T) {
// CHECK:STDOUT:   %T.loc15_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc15_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%T, constants.%Self.3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %.loc16_20.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T.loc15_22.2) {
// CHECK:STDOUT:   %T.loc15_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc15_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%.11) {
// CHECK:STDOUT:   %Self => constants.%.11
// CHECK:STDOUT:   %.loc8_15.1 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.4
// CHECK:STDOUT:   %Convert => constants.%Convert.4
// CHECK:STDOUT:   %.loc12_32.2 => constants.%.13
// CHECK:STDOUT:   %.loc12_32.3 => constants.%.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%i32, constants.%.15) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT:   %Self => constants.%.15
// CHECK:STDOUT:   %.loc12_20.1 => Core.IntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32) {
// CHECK:STDOUT:   %T.loc15_22.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc15_22.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self.3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.6
// CHECK:STDOUT:   %Convert => constants.%Convert.6
// CHECK:STDOUT:   %.loc16_32.2 => constants.%.17
// CHECK:STDOUT:   %.loc16_32.3 => constants.%.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%i32, constants.%.19) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%.19
// CHECK:STDOUT:   %.loc16_20.1 => Core.IntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc15_22.2 => Core.IntLiteral
// CHECK:STDOUT:   %T.patt.loc15_22.2 => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self.2 => constants.%Self.3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.8
// CHECK:STDOUT:   %Convert => constants.%Convert.8
// CHECK:STDOUT:   %.loc16_32.2 => constants.%.21
// CHECK:STDOUT:   %.loc16_32.3 => constants.%.22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(Core.IntLiteral, constants.%.23) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%.23
// CHECK:STDOUT:   %.loc16_20.1 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %As.type.2: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1: @As.%As.type (%As.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.2: %As.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @As(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = facet_access_type %Self.2 [symbolic]
// CHECK:STDOUT:   %.4: type = assoc_entity_type %As.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.5: %.4 = assoc_entity element0, imports.%import_ref.6 [symbolic]
// CHECK:STDOUT:   %.6: type = facet_access_type %Self.1 [symbolic]
// CHECK:STDOUT:   %As.type.3: type = facet_type <@As, @As(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @As(%i32) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %As.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %.9: %.4 = assoc_entity element0, imports.%import_ref.7 [symbolic]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [template]
// CHECK:STDOUT:   %Self.3: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.4: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Self.5: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %.10: type = facet_access_type %Self.5 [symbolic]
// CHECK:STDOUT:   %.11: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.3 [symbolic]
// CHECK:STDOUT:   %.12: %.11 = assoc_entity element0, imports.%import_ref.23 [symbolic]
// CHECK:STDOUT:   %.13: type = facet_access_type %Self.4 [symbolic]
// CHECK:STDOUT:   %Convert.type.4: type = fn_type @Convert.2, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.4: %Convert.type.4 = struct_value () [template]
// CHECK:STDOUT:   %.14: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.4 [template]
// CHECK:STDOUT:   %.15: %.14 = assoc_entity element0, imports.%import_ref.24 [template]
// CHECK:STDOUT:   %ImplicitAs.type.4: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.2, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.5: %Convert.type.5 = struct_value () [template]
// CHECK:STDOUT:   %.16: type = assoc_entity_type %ImplicitAs.type.4, %Convert.type.5 [template]
// CHECK:STDOUT:   %.17: %.16 = assoc_entity element0, imports.%import_ref.28 [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.6: %Convert.type.6 = struct_value () [template]
// CHECK:STDOUT:   %.18: <witness> = interface_witness (%Convert.6) [template]
// CHECK:STDOUT:   %.19: <bound method> = bound_method %.2, %Convert.6 [template]
// CHECK:STDOUT:   %.20: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.21: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.22: <bound method> = bound_method %.21, %Convert.6 [template]
// CHECK:STDOUT:   %.23: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %.24: type = facet_access_type %Self.3 [symbolic]
// CHECK:STDOUT:   %.25: type = assoc_entity_type %Add.type, %Op.type.1 [template]
// CHECK:STDOUT:   %.26: %.25 = assoc_entity element0, imports.%import_ref.30 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.20, %Op.2 [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %.30: %.11 = assoc_entity element0, imports.%import_ref.32 [symbolic]
// CHECK:STDOUT:   %Convert.type.7: type = fn_type @Convert.4 [template]
// CHECK:STDOUT:   %Convert.7: %Convert.type.7 = struct_value () [template]
// CHECK:STDOUT:   %.31: <witness> = interface_witness (%Convert.7) [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.29, %Convert.7 [template]
// CHECK:STDOUT:   %.33: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.34: type = array_type %.33, %i32 [template]
// CHECK:STDOUT:   %.36: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.37: <bound method> = bound_method %.33, %Convert.6 [template]
// CHECK:STDOUT:   %.38: <bound method> = bound_method %.36, %Convert.6 [template]
// CHECK:STDOUT:   %.39: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %.40: <bound method> = bound_method %.29, %Op.2 [template]
// CHECK:STDOUT:   %.41: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, %i32) [template]
// CHECK:STDOUT:   %.42: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.8: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.8: %Convert.type.8 = struct_value () [template]
// CHECK:STDOUT:   %.43: <witness> = interface_witness (%Convert.8) [template]
// CHECK:STDOUT:   %.44: <bound method> = bound_method %.33, %Convert.8 [template]
// CHECK:STDOUT:   %.45: <bound method> = bound_method %.36, %Convert.8 [template]
// CHECK:STDOUT:   %array: %.34 = tuple_value (%.29, %.39, %.41) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .As = %import_ref.2
// CHECK:STDOUT:     .Add = %import_ref.29
// CHECK:STDOUT:     .ImplicitAs = %import_ref.31
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//default, inst+70, unloaded
// CHECK:STDOUT:   %import_ref.4: @As.%.1 (%.4) = import_ref Core//default, inst+94, loaded [symbolic = @As.%.2 (constants.%.9)]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//default, inst+86, unloaded
// CHECK:STDOUT:   %import_ref.6 = import_ref Core//default, inst+86, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Core//default, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.9: %.25 = import_ref Core//default, inst+56, loaded [template = constants.%.26]
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//default, inst+50, unloaded
// CHECK:STDOUT:   %import_ref.11: type = import_ref Core//default, inst+149, loaded [template = constants.%i32]
// CHECK:STDOUT:   %import_ref.12: type = import_ref Core//default, inst+150, loaded [template = constants.%Add.type]
// CHECK:STDOUT:   %import_ref.13: <witness> = import_ref Core//default, inst+171, loaded [template = constants.%.27]
// CHECK:STDOUT:   %import_ref.14: type = import_ref Core//default, inst+176, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.15: type = import_ref Core//default, inst+182, loaded [template = constants.%As.type.3]
// CHECK:STDOUT:   %import_ref.16: <witness> = import_ref Core//default, inst+206, loaded [template = constants.%.18]
// CHECK:STDOUT:   %import_ref.17 = import_ref Core//default, inst+112, unloaded
// CHECK:STDOUT:   %import_ref.18: @ImplicitAs.%.1 (%.11) = import_ref Core//default, inst+136, loaded [symbolic = @ImplicitAs.%.2 (constants.%.30)]
// CHECK:STDOUT:   %import_ref.19 = import_ref Core//default, inst+128, unloaded
// CHECK:STDOUT:   %import_ref.20: type = import_ref Core//default, inst+211, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.21: type = import_ref Core//default, inst+217, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.22: <witness> = import_ref Core//default, inst+241, loaded [template = constants.%.43]
// CHECK:STDOUT:   %import_ref.23 = import_ref Core//default, inst+128, unloaded
// CHECK:STDOUT:   %import_ref.25: type = import_ref Core//default, inst+246, loaded [template = constants.%i32]
// CHECK:STDOUT:   %import_ref.26: type = import_ref Core//default, inst+252, loaded [template = constants.%ImplicitAs.type.4]
// CHECK:STDOUT:   %import_ref.27: <witness> = import_ref Core//default, inst+276, loaded [template = constants.%.31]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc4_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc4_11: init type = call constants.%Int(%.loc4_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_17: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc4_22: init type = call constants.%Int(%.loc4_22.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_22.2: type = value_of_initializer %int.make_type_signed.loc4_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_22.3: type = converted %int.make_type_signed.loc4_22, %.loc4_22.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_19.1: %Convert.type.2 = interface_witness_access constants.%.18, element0 [template = constants.%Convert.6]
// CHECK:STDOUT:   %.loc4_19.2: <bound method> = bound_method %.loc4_17, %.loc4_19.1 [template = constants.%.19]
// CHECK:STDOUT:   %int.convert_checked.loc4_19: init %i32 = call %.loc4_19.2(%.loc4_17) [template = constants.%.20]
// CHECK:STDOUT:   %.loc4_19.3: %i32 = value_of_initializer %int.convert_checked.loc4_19 [template = constants.%.20]
// CHECK:STDOUT:   %.loc4_19.4: %i32 = converted %.loc4_17, %.loc4_19.3 [template = constants.%.20]
// CHECK:STDOUT:   %.loc4_30: Core.IntLiteral = int_value 2 [template = constants.%.21]
// CHECK:STDOUT:   %.loc4_35.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc4_35: init type = call constants.%Int(%.loc4_35.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_35.2: type = value_of_initializer %int.make_type_signed.loc4_35 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_35.3: type = converted %int.make_type_signed.loc4_35, %.loc4_35.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_32.1: %Convert.type.2 = interface_witness_access constants.%.18, element0 [template = constants.%Convert.6]
// CHECK:STDOUT:   %.loc4_32.2: <bound method> = bound_method %.loc4_30, %.loc4_32.1 [template = constants.%.22]
// CHECK:STDOUT:   %int.convert_checked.loc4_32: init %i32 = call %.loc4_32.2(%.loc4_30) [template = constants.%.23]
// CHECK:STDOUT:   %.loc4_32.3: %i32 = value_of_initializer %int.convert_checked.loc4_32 [template = constants.%.23]
// CHECK:STDOUT:   %.loc4_32.4: %i32 = converted %.loc4_30, %.loc4_32.3 [template = constants.%.23]
// CHECK:STDOUT:   %.loc4_27.1: %Op.type.1 = interface_witness_access constants.%.27, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %.loc4_27.2: <bound method> = bound_method %.loc4_19.4, %.loc4_27.1 [template = constants.%.28]
// CHECK:STDOUT:   %int.sadd: init %i32 = call %.loc4_27.2(%.loc4_19.4, %.loc4_32.4) [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_11.2: type = value_of_initializer %int.make_type_signed.loc4_11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_11.3: type = converted %int.make_type_signed.loc4_11, %.loc4_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_27.3: %Convert.type.5 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.7]
// CHECK:STDOUT:   %.loc4_27.4: <bound method> = bound_method %int.sadd, %.loc4_27.3 [template = constants.%.32]
// CHECK:STDOUT:   %.loc4_27.5: %i32 = value_of_initializer %int.sadd [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_27.6: %i32 = converted %int.sadd, %.loc4_27.5 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc4_27: init Core.IntLiteral = call %.loc4_27.4(%.loc4_27.6) [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_27.7: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_27 [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_27.8: Core.IntLiteral = converted %int.sadd, %.loc4_27.7 [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_39: type = array_type %.loc4_27.8, %i32 [template = constants.%.34]
// CHECK:STDOUT:   %arr.var: ref %.34 = var arr
// CHECK:STDOUT:   %arr: ref %.34 = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%T) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @As.%As.type (%As.type.2), @As.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %.2: @As.%.1 (%.4) = assoc_entity element0, imports.%import_ref.6 [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.3
// CHECK:STDOUT:     .Convert = imports.%import_ref.4
// CHECK:STDOUT:     witness = (imports.%import_ref.5)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8
// CHECK:STDOUT:   .Op = imports.%import_ref.9
// CHECK:STDOUT:   witness = (imports.%import_ref.10)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.5)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.3)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.3) = struct_value () [symbolic = %Convert (constants.%Convert.3)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.3) [symbolic = %.1 (constants.%.11)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.11) = assoc_entity element0, imports.%import_ref.23 [symbolic = %.2 (constants.%.12)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.17
// CHECK:STDOUT:     .Convert = imports.%import_ref.18
// CHECK:STDOUT:     witness = (imports.%import_ref.19)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.11 as imports.%import_ref.12 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.14 as imports.%import_ref.15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%import_ref.20 as imports.%import_ref.21 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.25 as imports.%import_ref.26 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%T: type, constants.%Self.1: @As.%As.type (%As.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.2)]
// CHECK:STDOUT:   %Self: %As.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic = %.1 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%.1 (%.3)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%T: type, constants.%Self.4: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.5)]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic = %.1 (constants.%.10)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%.1 (%.10)]() -> @Convert.2.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: Core.IntLiteral]() -> %i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(constants.%Self.3: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %.1: type = facet_access_type %Self [symbolic = %.1 (constants.%.24)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%.1 (%.24)](%other.param_patt: @Op.1.%.1 (%.24)) -> @Op.1.%.1 (%.24);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self.param_patt: %i32](%other.param_patt: %i32) -> %i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: %i32]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: Core.IntLiteral]() -> %i32 = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_44: Core.IntLiteral = int_value 3 [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_47: Core.IntLiteral = int_value 4 [template = constants.%.36]
// CHECK:STDOUT:   %.loc4_51: Core.IntLiteral = int_value 3 [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_56.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc4_56: init type = call constants.%Int(%.loc4_56.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_56.2: type = value_of_initializer %int.make_type_signed.loc4_56 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_56.3: type = converted %int.make_type_signed.loc4_56, %.loc4_56.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_53.1: %Convert.type.2 = interface_witness_access constants.%.18, element0 [template = constants.%Convert.6]
// CHECK:STDOUT:   %.loc4_53.2: <bound method> = bound_method %.loc4_51, %.loc4_53.1 [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc4_53: init %i32 = call %.loc4_53.2(%.loc4_51) [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_53.3: %i32 = value_of_initializer %int.convert_checked.loc4_53 [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_53.4: %i32 = converted %.loc4_51, %.loc4_53.3 [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_64: Core.IntLiteral = int_value 4 [template = constants.%.36]
// CHECK:STDOUT:   %.loc4_69.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc4_69: init type = call constants.%Int(%.loc4_69.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_69.2: type = value_of_initializer %int.make_type_signed.loc4_69 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_69.3: type = converted %int.make_type_signed.loc4_69, %.loc4_69.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_66.1: %Convert.type.2 = interface_witness_access constants.%.18, element0 [template = constants.%Convert.6]
// CHECK:STDOUT:   %.loc4_66.2: <bound method> = bound_method %.loc4_64, %.loc4_66.1 [template = constants.%.38]
// CHECK:STDOUT:   %int.convert_checked.loc4_66: init %i32 = call %.loc4_66.2(%.loc4_64) [template = constants.%.39]
// CHECK:STDOUT:   %.loc4_66.3: %i32 = value_of_initializer %int.convert_checked.loc4_66 [template = constants.%.39]
// CHECK:STDOUT:   %.loc4_66.4: %i32 = converted %.loc4_64, %.loc4_66.3 [template = constants.%.39]
// CHECK:STDOUT:   %.loc4_61.1: %Op.type.1 = interface_witness_access constants.%.27, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %.loc4_61.2: <bound method> = bound_method %.loc4_53.4, %.loc4_61.1 [template = constants.%.40]
// CHECK:STDOUT:   %int.sadd: init %i32 = call %.loc4_61.2(%.loc4_53.4, %.loc4_66.4) [template = constants.%.41]
// CHECK:STDOUT:   %.loc4_73.1: %tuple.type = tuple_literal (%.loc4_44, %.loc4_47, %int.sadd)
// CHECK:STDOUT:   %.loc4_73.2: %Convert.type.4 = interface_witness_access constants.%.43, element0 [template = constants.%Convert.8]
// CHECK:STDOUT:   %.loc4_73.3: <bound method> = bound_method %.loc4_44, %.loc4_73.2 [template = constants.%.44]
// CHECK:STDOUT:   %int.convert_checked.loc4_73.1: init %i32 = call %.loc4_73.3(%.loc4_44) [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_73.4: init %i32 = converted %.loc4_44, %int.convert_checked.loc4_73.1 [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_73.5: Core.IntLiteral = int_value 0 [template = constants.%.42]
// CHECK:STDOUT:   %.loc4_73.6: ref %i32 = array_index file.%arr.var, %.loc4_73.5
// CHECK:STDOUT:   %.loc4_73.7: init %i32 = initialize_from %.loc4_73.4 to %.loc4_73.6 [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_73.8: %Convert.type.4 = interface_witness_access constants.%.43, element0 [template = constants.%Convert.8]
// CHECK:STDOUT:   %.loc4_73.9: <bound method> = bound_method %.loc4_47, %.loc4_73.8 [template = constants.%.45]
// CHECK:STDOUT:   %int.convert_checked.loc4_73.2: init %i32 = call %.loc4_73.9(%.loc4_47) [template = constants.%.39]
// CHECK:STDOUT:   %.loc4_73.10: init %i32 = converted %.loc4_47, %int.convert_checked.loc4_73.2 [template = constants.%.39]
// CHECK:STDOUT:   %.loc4_73.11: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_73.12: ref %i32 = array_index file.%arr.var, %.loc4_73.11
// CHECK:STDOUT:   %.loc4_73.13: init %i32 = initialize_from %.loc4_73.10 to %.loc4_73.12 [template = constants.%.39]
// CHECK:STDOUT:   %.loc4_73.14: Core.IntLiteral = int_value 2 [template = constants.%.21]
// CHECK:STDOUT:   %.loc4_73.15: ref %i32 = array_index file.%arr.var, %.loc4_73.14
// CHECK:STDOUT:   %.loc4_73.16: init %i32 = initialize_from %int.sadd to %.loc4_73.15 [template = constants.%.41]
// CHECK:STDOUT:   %.loc4_73.17: init %.34 = array_init (%.loc4_73.7, %.loc4_73.13, %.loc4_73.16) to file.%arr.var [template = constants.%array]
// CHECK:STDOUT:   %.loc4_74: init %.34 = converted %.loc4_73.1, %.loc4_73.17 [template = constants.%array]
// CHECK:STDOUT:   assign file.%arr.var, %.loc4_74
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.2
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT:   %.1 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.3
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %.2 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.5
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.4
// CHECK:STDOUT:   %Convert => constants.%Convert.4
// CHECK:STDOUT:   %.1 => constants.%.14
// CHECK:STDOUT:   %.2 => constants.%.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%T, constants.%Self.4) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.4
// CHECK:STDOUT:   %.1 => constants.%.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %T.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.4
// CHECK:STDOUT:   %Self => constants.%Self.5
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.5
// CHECK:STDOUT:   %Convert => constants.%Convert.5
// CHECK:STDOUT:   %.1 => constants.%.16
// CHECK:STDOUT:   %.2 => constants.%.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.3) {
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %.1 => constants.%.24
// CHECK:STDOUT: }
// CHECK:STDOUT:

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

fn Int32() -> type = "int.make_type_32";
fn IntLiteral() -> type = "int_literal.make_type";

interface Add {
  fn Op[self: Self](other: Self) -> Self;
}

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

impl i32 as Add {
  fn Op[self: Self](other: Self) -> Self = "int.sadd";
}

impl i32 as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// --- user.carbon

import Core;

var arr: [i32; 1 + 2] = (3, 4, 3 + 4);

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [template]
// CHECK:STDOUT:   %Self.1: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.1: %Op.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %Add.type, %Op.type.1 [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @Add.%Op.decl [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.1: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs: %ImplicitAs.type.1 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.3 [template]
// CHECK:STDOUT:   %.7: %.6 = assoc_entity element0, @ImplicitAs.%Convert.decl [template]
// CHECK:STDOUT:   %.8: <witness> = interface_witness (%Convert.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int32 = %Int32.decl
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int32.decl: %Int32.type = fn_decl @Int32 [template = constants.%Int32] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [template = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.decl: type = interface_decl @Add [template = constants.%Add.type] {} {}
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.1 = interface_decl @ImplicitAs [template = constants.%ImplicitAs] {
// CHECK:STDOUT:     %T.patt.loc11_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_22.1, runtime_param<invalid> [symbolic = %T.patt.loc11_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_22.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_6.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc15_6.2: type = converted %int.make_type_32, %.loc15_6.1 [template = i32]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, file.%Add.decl [template = constants.%Add.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc19_6.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc19_6.2: type = converted %int.make_type_32, %.loc19_6.1 [template = i32]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.1 = name_ref ImplicitAs, file.%ImplicitAs.decl [template = constants.%ImplicitAs]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc19_23.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc19_23.2: type = converted %int_literal.make_type, %.loc19_23.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %Op.decl: %Op.type.1 = fn_decl @Op.1 [template = constants.%Op.1] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self (%Self.1) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self (%Self.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%Self (%Self.1) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%Self (%Self.1) = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%Self (%Self.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%Self (%Self.1) = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc8_15: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc8_15.1: type = facet_type_access %Self.ref.loc8_15 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc8_15.2: type = converted %Self.ref.loc8_15, %.loc8_15.1 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %Self.ref.loc8_28: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc8_28.1: type = facet_type_access %Self.ref.loc8_28 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc8_28.2: type = converted %Self.ref.loc8_28, %.loc8_28.1 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %Self.ref.loc8_37: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc8_37.1: type = facet_type_access %Self.ref.loc8_37 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %.loc8_37.2: type = converted %Self.ref.loc8_37, %.loc8_37.1 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self (%Self.1) = value_param runtime_param0
// CHECK:STDOUT:     %self: @Op.1.%Self (%Self.1) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self (%Self.1) = value_param runtime_param1
// CHECK:STDOUT:     %other: @Op.1.%Self (%Self.1) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self (%Self.1) = out_param runtime_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self (%Self.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: %.1 = assoc_entity element0, %Op.decl [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %.loc8
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc11_22.1: type) {
// CHECK:STDOUT:   %T.loc11_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc11_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T.loc11_22.2) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.loc12_32.2: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.loc12_32.2 (constants.%.3)]
// CHECK:STDOUT:   %.loc12_32.3: @ImplicitAs.%.loc12_32.2 (%.3) = assoc_entity element0, %Convert.decl [symbolic = %.loc12_32.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.2)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.1) = fn_decl @Convert.1 [symbolic = @ImplicitAs.%Convert (constants.%Convert.1)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self (%Self.2) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self (%Self.2) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc12_20.1: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.2) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:       %Self.ref: @Convert.1.%ImplicitAs.type (%ImplicitAs.type.2) = name_ref Self, %.loc12_20.1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:       %.loc12_20.2: type = facet_type_access %Self.ref [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:       %.loc12_20.3: type = converted %Self.ref, %.loc12_20.2 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc11_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self (%Self.2) = value_param runtime_param0
// CHECK:STDOUT:       %self: @Convert.1.%Self (%Self.2) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc12_32.1: @ImplicitAs.%.loc12_32.2 (%.3) = assoc_entity element0, %Convert.decl [symbolic = %.loc12_32.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Convert = %.loc12_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc15_6.2 as %Add.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.2 = fn_decl @Op.2 [template = constants.%Op.2] {
// CHECK:STDOUT:     %self.patt: i32 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: i32 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: i32 = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: i32 = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc16_15: type = name_ref Self, @impl.1.%.loc15_6.2 [template = i32]
// CHECK:STDOUT:     %Self.ref.loc16_28: type = name_ref Self, @impl.1.%.loc15_6.2 [template = i32]
// CHECK:STDOUT:     %Self.ref.loc16_37: type = name_ref Self, @impl.1.%.loc15_6.2 [template = i32]
// CHECK:STDOUT:     %self.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %self: i32 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: i32 = value_param runtime_param1
// CHECK:STDOUT:     %other: i32 = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_17: <witness> = interface_witness (%Op.decl) [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = %.loc15_17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %.loc19_6.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.2 = fn_decl @Convert.2 [template = constants.%Convert.2] {
// CHECK:STDOUT:     %self.patt: i32 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: i32 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.2.%.loc19_6.2 [template = i32]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc20_42.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc20_42.2: type = converted %int_literal.make_type, %.loc20_42.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %self.param: i32 = value_param runtime_param0
// CHECK:STDOUT:     %self: i32 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_38: <witness> = interface_witness (%Convert.decl) [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = %.loc19_38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@Add.%Self: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self (%Self.1)](%other.param_patt: @Op.1.%Self (%Self.1)) -> @Op.1.%Self (%Self.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@ImplicitAs.%T.loc11_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.2)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self.param_patt: i32](%other.param_patt: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: i32]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc11_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {
// CHECK:STDOUT:   %T.loc11_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.2) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T.loc11_22.2) {
// CHECK:STDOUT:   %T.loc11_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(i32) {
// CHECK:STDOUT:   %Self => i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc11_22.2 => Core.IntLiteral
// CHECK:STDOUT:   %T.patt.loc11_22.2 => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self.2 => constants.%Self.2
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.3
// CHECK:STDOUT:   %Convert => constants.%Convert.3
// CHECK:STDOUT:   %.loc12_32.2 => constants.%.6
// CHECK:STDOUT:   %.loc12_32.3 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(Core.IntLiteral, i32) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_value 2 [template]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [template]
// CHECK:STDOUT:   %Self.1: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %Add.type, %Op.type.1 [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, imports.%import_ref.6 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2) = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Self.3: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Convert.type.1: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.1: %Convert.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.5: type = assoc_entity_type %ImplicitAs.type.2, %Convert.type.1 [symbolic]
// CHECK:STDOUT:   %.6: %.5 = assoc_entity element0, imports.%import_ref.16 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.2: %Convert.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type %ImplicitAs.type.3, %Convert.type.2 [template]
// CHECK:STDOUT:   %.8: %.7 = assoc_entity element0, imports.%import_ref.17 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.9: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %.10: <bound method> = bound_method %.1, %Op.2 [template]
// CHECK:STDOUT:   %.11: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.12: %.5 = assoc_entity element0, imports.%import_ref.19 [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2 [template]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.13: <witness> = interface_witness (%Convert.3) [template]
// CHECK:STDOUT:   %.14: <bound method> = bound_method %.11, %Convert.3 [template]
// CHECK:STDOUT:   %.15: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.16: type = array_type %.15, i32 [template]
// CHECK:STDOUT:   %.18: i32 = int_value 4 [template]
// CHECK:STDOUT:   %.19: <bound method> = bound_method %.11, %Op.2 [template]
// CHECK:STDOUT:   %.20: i32 = int_value 7 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.21: i32 = int_value 0 [template]
// CHECK:STDOUT:   %array: %.16 = tuple_value (%.11, %.18, %.20) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .Add = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.18
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3 = import_ref Core//default, inst+18, unloaded
// CHECK:STDOUT:   %import_ref.4: %.3 = import_ref Core//default, inst+46, loaded [template = constants.%.4]
// CHECK:STDOUT:   %import_ref.5 = import_ref Core//default, inst+41, unloaded
// CHECK:STDOUT:   %import_ref.7: type = import_ref Core//default, inst+92, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.8: type = import_ref Core//default, inst+93, loaded [template = constants.%Add.type]
// CHECK:STDOUT:   %import_ref.9: <witness> = import_ref Core//default, inst+113, loaded [template = constants.%.9]
// CHECK:STDOUT:   %import_ref.10 = import_ref Core//default, inst+60, unloaded
// CHECK:STDOUT:   %import_ref.11: @ImplicitAs.%.1 (%.5) = import_ref Core//default, inst+82, loaded [symbolic = @ImplicitAs.%.2 (constants.%.12)]
// CHECK:STDOUT:   %import_ref.12 = import_ref Core//default, inst+75, unloaded
// CHECK:STDOUT:   %import_ref.13: type = import_ref Core//default, inst+117, loaded [template = i32]
// CHECK:STDOUT:   %import_ref.14: type = import_ref Core//default, inst+123, loaded [template = constants.%ImplicitAs.type.3]
// CHECK:STDOUT:   %import_ref.15: <witness> = import_ref Core//default, inst+146, loaded [template = constants.%.13]
// CHECK:STDOUT:   %import_ref.16 = import_ref Core//default, inst+75, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc4_16: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_20: i32 = int_value 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_18.1: %Op.type.1 = interface_witness_access constants.%.9, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %.loc4_18.2: <bound method> = bound_method %.loc4_16, %.loc4_18.1 [template = constants.%.10]
// CHECK:STDOUT:   %int.sadd: init i32 = call %.loc4_18.2(%.loc4_16, %.loc4_20) [template = constants.%.11]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_32, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:   %.loc4_18.3: %Convert.type.2 = interface_witness_access constants.%.13, element0 [template = constants.%Convert.3]
// CHECK:STDOUT:   %.loc4_18.4: <bound method> = bound_method %int.sadd, %.loc4_18.3 [template = constants.%.14]
// CHECK:STDOUT:   %.loc4_18.5: i32 = value_of_initializer %int.sadd [template = constants.%.11]
// CHECK:STDOUT:   %.loc4_18.6: i32 = converted %int.sadd, %.loc4_18.5 [template = constants.%.11]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %.loc4_18.4(%.loc4_18.6) [template = constants.%.15]
// CHECK:STDOUT:   %.loc4_18.7: Core.IntLiteral = value_of_initializer %int.convert_checked [template = constants.%.15]
// CHECK:STDOUT:   %.loc4_18.8: Core.IntLiteral = converted %int.sadd, %.loc4_18.7 [template = constants.%.15]
// CHECK:STDOUT:   %.loc4_21: type = array_type %.loc4_18.8, i32 [template = constants.%.16]
// CHECK:STDOUT:   %arr.var: ref %.16 = var arr
// CHECK:STDOUT:   %arr: ref %.16 = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .Op = imports.%import_ref.4
// CHECK:STDOUT:   witness = (imports.%import_ref.5)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%T: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.1)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.1) = struct_value () [symbolic = %Convert (constants.%Convert.1)]
// CHECK:STDOUT:   %.1: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2), @ImplicitAs.%Convert.type (%Convert.type.1) [symbolic = %.1 (constants.%.5)]
// CHECK:STDOUT:   %.2: @ImplicitAs.%.1 (%.5) = assoc_entity element0, imports.%import_ref.16 [symbolic = %.2 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.10
// CHECK:STDOUT:     .Convert = imports.%import_ref.11
// CHECK:STDOUT:     witness = (imports.%import_ref.12)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.7 as imports.%import_ref.8 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.13 as imports.%import_ref.14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(constants.%Self.1: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self (%Self.1)](%other.param_patt: @Op.1.%Self (%Self.1)) -> @Op.1.%Self (%Self.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%T: type, constants.%Self.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self (%Self.3)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self.param_patt: i32](%other.param_patt: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: i32]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_26: i32 = int_value 3 [template = constants.%.11]
// CHECK:STDOUT:   %.loc4_29: i32 = int_value 4 [template = constants.%.18]
// CHECK:STDOUT:   %.loc4_32: i32 = int_value 3 [template = constants.%.11]
// CHECK:STDOUT:   %.loc4_36: i32 = int_value 4 [template = constants.%.18]
// CHECK:STDOUT:   %.loc4_34.1: %Op.type.1 = interface_witness_access constants.%.9, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %.loc4_34.2: <bound method> = bound_method %.loc4_32, %.loc4_34.1 [template = constants.%.19]
// CHECK:STDOUT:   %int.sadd: init i32 = call %.loc4_34.2(%.loc4_32, %.loc4_36) [template = constants.%.20]
// CHECK:STDOUT:   %.loc4_37.1: %tuple.type = tuple_literal (%.loc4_26, %.loc4_29, %int.sadd)
// CHECK:STDOUT:   %.loc4_37.2: i32 = int_value 0 [template = constants.%.21]
// CHECK:STDOUT:   %.loc4_37.3: ref i32 = array_index file.%arr.var, %.loc4_37.2
// CHECK:STDOUT:   %.loc4_37.4: init i32 = initialize_from %.loc4_26 to %.loc4_37.3 [template = constants.%.11]
// CHECK:STDOUT:   %.loc4_37.5: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_37.6: ref i32 = array_index file.%arr.var, %.loc4_37.5
// CHECK:STDOUT:   %.loc4_37.7: init i32 = initialize_from %.loc4_29 to %.loc4_37.6 [template = constants.%.18]
// CHECK:STDOUT:   %.loc4_37.8: i32 = int_value 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_37.9: ref i32 = array_index file.%arr.var, %.loc4_37.8
// CHECK:STDOUT:   %.loc4_37.10: init i32 = initialize_from %int.sadd to %.loc4_37.9 [template = constants.%.20]
// CHECK:STDOUT:   %.loc4_37.11: init %.16 = array_init (%.loc4_37.4, %.loc4_37.7, %.loc4_37.10) to file.%arr.var [template = constants.%array]
// CHECK:STDOUT:   %.loc4_38: init %.16 = converted %.loc4_37.1, %.loc4_37.11 [template = constants.%array]
// CHECK:STDOUT:   assign file.%arr.var, %.loc4_38
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %T.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3
// CHECK:STDOUT:   %Self => constants.%Self.3
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2
// CHECK:STDOUT:   %Convert => constants.%Convert.2
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT:   %.2 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.1.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.2) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.2
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

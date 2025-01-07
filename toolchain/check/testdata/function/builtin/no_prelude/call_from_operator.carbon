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
// CHECK:STDOUT:   %Self.d51f: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type.c8e7: type = facet_access_type %Self.d51f [symbolic]
// CHECK:STDOUT:   %Op.type.16a5: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.93e6: %Op.type.16a5 = struct_value () [template]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type %Add.type, %Op.type.16a5 [template]
// CHECK:STDOUT:   %assoc0.dcfa: %Op.assoc_type = assoc_entity element0, @Add.%Op.decl [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %As.type.b510: type = generic_interface_type @As [template]
// CHECK:STDOUT:   %As.generic: %As.type.b510 = struct_value () [template]
// CHECK:STDOUT:   %As.type.8b04: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.a390: %As.type.8b04 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.5fac: type = facet_access_type %Self.a390 [symbolic]
// CHECK:STDOUT:   %Convert.type.412e: type = fn_type @Convert.1, @As(%T) [symbolic]
// CHECK:STDOUT:   %Convert.adcb: %Convert.type.412e = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.d24a: type = assoc_entity_type %As.type.8b04, %Convert.type.412e [symbolic]
// CHECK:STDOUT:   %assoc0.e2f7: %Convert.assoc_type.d24a = assoc_entity element0, @As.%Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.96f9: type = generic_interface_type @ImplicitAs [template]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.96f9 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.a1f2: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.8196: %ImplicitAs.type.a1f2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.dc3d: type = facet_access_type %Self.8196 [symbolic]
// CHECK:STDOUT:   %Convert.type.b111: type = fn_type @Convert.2, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.639f: %Convert.type.b111 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.4d84: type = assoc_entity_type %ImplicitAs.type.a1f2, %Convert.type.b111 [symbolic]
// CHECK:STDOUT:   %assoc0.39c2: %Convert.assoc_type.4d84 = assoc_entity element0, @ImplicitAs.%Convert.decl [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %impl_witness.8dc7: <witness> = impl_witness (@impl.1.%Op.decl) [template]
// CHECK:STDOUT:   %Op.type.9ef5: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.11af: %Op.type.9ef5 = struct_value () [template]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %i32.builtin, %impl_witness.8dc7 [template]
// CHECK:STDOUT:   %As.type.8e7e: type = facet_type <@As, @As(%i32.builtin)> [template]
// CHECK:STDOUT:   %Convert.type.d050: type = fn_type @Convert.1, @As(%i32.builtin) [template]
// CHECK:STDOUT:   %Convert.5e5c: %Convert.type.d050 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.5961: type = assoc_entity_type %As.type.8e7e, %Convert.type.d050 [template]
// CHECK:STDOUT:   %assoc0.c6b6: %Convert.assoc_type.5961 = assoc_entity element0, @As.%Convert.decl [template]
// CHECK:STDOUT:   %impl_witness.fcea: <witness> = impl_witness (@impl.2.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.6e53: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.b6a5: %Convert.type.6e53 = struct_value () [template]
// CHECK:STDOUT:   %As.facet: %As.type.8b04 = facet_value Core.IntLiteral, %impl_witness.fcea [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.9821: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [template]
// CHECK:STDOUT:   %Convert.type.c1fa: type = fn_type @Convert.2, @ImplicitAs(%i32.builtin) [template]
// CHECK:STDOUT:   %Convert.3381: %Convert.type.c1fa = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.9185: type = assoc_entity_type %ImplicitAs.type.9821, %Convert.type.c1fa [template]
// CHECK:STDOUT:   %assoc0.e51d: %Convert.assoc_type.9185 = assoc_entity element0, @ImplicitAs.%Convert.decl [template]
// CHECK:STDOUT:   %impl_witness.c5ef: <witness> = impl_witness (@impl.3.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.9280: type = fn_type @Convert.4 [template]
// CHECK:STDOUT:   %Convert.4e26: %Convert.type.9280 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.f011: %ImplicitAs.type.a1f2 = facet_value Core.IntLiteral, %impl_witness.c5ef [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5f72: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.ddef: type = fn_type @Convert.2, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.254f: %Convert.type.ddef = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.6c13: type = assoc_entity_type %ImplicitAs.type.5f72, %Convert.type.ddef [template]
// CHECK:STDOUT:   %assoc0.143a: %Convert.assoc_type.6c13 = assoc_entity element0, @ImplicitAs.%Convert.decl [template]
// CHECK:STDOUT:   %impl_witness.2ba2: <witness> = impl_witness (@impl.4.%Convert.decl) [template]
// CHECK:STDOUT:   %Convert.type.5789: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.5d6f: %Convert.type.5789 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet.f95c: %ImplicitAs.type.a1f2 = facet_value %i32.builtin, %impl_witness.2ba2 [symbolic]
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
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc5_22.1: type = splice_block %.loc5_22.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_22.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_22.3: type = converted %int_literal.make_type, %.loc5_22.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.decl: type = interface_decl @Add [template = constants.%Add.type] {} {}
// CHECK:STDOUT:   %As.decl: %As.type.b510 = interface_decl @As [template = constants.%As.generic] {
// CHECK:STDOUT:     %T.patt.loc11_14.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_14.1, runtime_param<invalid> [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_14.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.96f9 = interface_decl @ImplicitAs [template = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt.loc15_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc15_22.1, runtime_param<invalid> [symbolic = %T.patt.loc15_22.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc15_22.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc15_22.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc19_6.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc19_6.2: type = converted %int.make_type_signed, %.loc19_6.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %Add.ref: type = name_ref Add, file.%Add.decl [template = constants.%Add.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc19: <witness> = impl_witness (@impl.1.%Op.decl) [template = constants.%impl_witness.8dc7]
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_17.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_17.2: type = converted %int_literal.make_type, %.loc23_17.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %As.ref: %As.type.b510 = name_ref As, file.%As.decl [template = constants.%As.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc23_28.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc23_28.2: type = converted %int.make_type_signed, %.loc23_28.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%i32.builtin)> [template = constants.%As.type.8e7e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc23: <witness> = impl_witness (@impl.2.%Convert.decl) [template = constants.%impl_witness.fcea]
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc27_17.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc27_17.2: type = converted %int_literal.make_type, %.loc27_17.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.96f9 = name_ref ImplicitAs, file.%ImplicitAs.decl [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc27_36.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc27_36.2: type = converted %int.make_type_signed, %.loc27_36.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32.builtin)> [template = constants.%ImplicitAs.type.9821]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc27: <witness> = impl_witness (@impl.3.%Convert.decl) [template = constants.%impl_witness.c5ef]
// CHECK:STDOUT:   impl_decl @impl.4 [template] {} {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc31_6.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc31_6.2: type = converted %int.make_type_signed, %.loc31_6.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.96f9 = name_ref ImplicitAs, file.%ImplicitAs.decl [template = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_36.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_36.2: type = converted %int_literal.make_type, %.loc31_36.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template = constants.%ImplicitAs.type.5f72]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc31: <witness> = impl_witness (@impl.4.%Convert.decl) [template = constants.%impl_witness.2ba2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.d51f]
// CHECK:STDOUT:   %Op.decl: %Op.type.16a5 = fn_decl @Op.1 [template = constants.%Op.93e6] {
// CHECK:STDOUT:     %self.patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7) = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7) = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7) = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc8_37: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.d51f)]
// CHECK:STDOUT:     %Self.as_type.loc8_37: type = facet_access_type %Self.ref.loc8_37 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c8e7)]
// CHECK:STDOUT:     %.loc8_37: type = converted %Self.ref.loc8_37, %Self.as_type.loc8_37 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c8e7)]
// CHECK:STDOUT:     %self.param: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7) = value_param runtime_param0
// CHECK:STDOUT:     %.loc8_15.1: type = splice_block %.loc8_15.2 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c8e7)] {
// CHECK:STDOUT:       %Self.ref.loc8_15: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.d51f)]
// CHECK:STDOUT:       %Self.as_type.loc8_15.2: type = facet_access_type %Self.ref.loc8_15 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c8e7)]
// CHECK:STDOUT:       %.loc8_15.2: type = converted %Self.ref.loc8_15, %Self.as_type.loc8_15.2 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c8e7)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7) = value_param runtime_param1
// CHECK:STDOUT:     %.loc8_28.1: type = splice_block %.loc8_28.2 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c8e7)] {
// CHECK:STDOUT:       %Self.ref.loc8_28: %Add.type = name_ref Self, @Add.%Self [symbolic = %Self (constants.%Self.d51f)]
// CHECK:STDOUT:       %Self.as_type.loc8_28: type = facet_access_type %Self.ref.loc8_28 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c8e7)]
// CHECK:STDOUT:       %.loc8_28.2: type = converted %Self.ref.loc8_28, %Self.as_type.loc8_28 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c8e7)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7) = out_param runtime_param2
// CHECK:STDOUT:     %return: ref @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %Op.assoc_type = assoc_entity element0, %Op.decl [template = constants.%assoc0.dcfa]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Op = %assoc0
// CHECK:STDOUT:   witness = (%Op.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%T.loc11_14.1: type) {
// CHECK:STDOUT:   %T.loc11_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_14.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_14.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_14.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T.loc11_14.2)> [symbolic = %As.type (constants.%As.type.8b04)]
// CHECK:STDOUT:   %Self.2: %As.type.8b04 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.a390)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%T.loc11_14.2) [symbolic = %Convert.type (constants.%Convert.type.412e)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.412e) = struct_value () [symbolic = %Convert (constants.%Convert.adcb)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.8b04), @As.%Convert.type (%Convert.type.412e) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.d24a)]
// CHECK:STDOUT:   %assoc0.loc12_32.2: @As.%Convert.assoc_type (%Convert.assoc_type.d24a) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.e2f7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @As.%As.type (%As.type.8b04) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.a390)]
// CHECK:STDOUT:     %Convert.decl: @As.%Convert.type (%Convert.type.412e) = fn_decl @Convert.1 [symbolic = @As.%Convert (constants.%Convert.adcb)] {
// CHECK:STDOUT:       %self.patt: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.5fac) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.5fac) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.1.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @As.%T.loc11_14.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.5fac) = value_param runtime_param0
// CHECK:STDOUT:       %.loc12_20.1: type = splice_block %.loc12_20.3 [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.5fac)] {
// CHECK:STDOUT:         %.loc12_20.2: @Convert.1.%As.type (%As.type.8b04) = specific_constant @As.%Self.1, @As(constants.%T) [symbolic = %Self (constants.%Self.a390)]
// CHECK:STDOUT:         %Self.ref: @Convert.1.%As.type (%As.type.8b04) = name_ref Self, %.loc12_20.2 [symbolic = %Self (constants.%Self.a390)]
// CHECK:STDOUT:         %Self.as_type.loc12_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.5fac)]
// CHECK:STDOUT:         %.loc12_20.3: type = converted %Self.ref, %Self.as_type.loc12_20.2 [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.5fac)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.5fac) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.1.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc12_32.1: @As.%Convert.assoc_type (%Convert.assoc_type.d24a) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.e2f7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Convert = %assoc0.loc12_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc15_22.1: type) {
// CHECK:STDOUT:   %T.loc15_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc15_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc15_22.2)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.a1f2)]
// CHECK:STDOUT:   %Self.2: %ImplicitAs.type.a1f2 = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.8196)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%T.loc15_22.2) [symbolic = %Convert.type (constants.%Convert.type.b111)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.b111) = struct_value () [symbolic = %Convert (constants.%Convert.639f)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.a1f2), @ImplicitAs.%Convert.type (%Convert.type.b111) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.4d84)]
// CHECK:STDOUT:   %assoc0.loc16_32.2: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.4d84) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc16_32.2 (constants.%assoc0.39c2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.a1f2) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.8196)]
// CHECK:STDOUT:     %Convert.decl: @ImplicitAs.%Convert.type (%Convert.type.b111) = fn_decl @Convert.2 [symbolic = @ImplicitAs.%Convert (constants.%Convert.639f)] {
// CHECK:STDOUT:       %self.patt: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.dc3d) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.dc3d) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @Convert.2.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @Convert.2.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc15_22.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.dc3d) = value_param runtime_param0
// CHECK:STDOUT:       %.loc16_20.1: type = splice_block %.loc16_20.3 [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.dc3d)] {
// CHECK:STDOUT:         %.loc16_20.2: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.a1f2) = specific_constant @ImplicitAs.%Self.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self.8196)]
// CHECK:STDOUT:         %Self.ref: @Convert.2.%ImplicitAs.type (%ImplicitAs.type.a1f2) = name_ref Self, %.loc16_20.2 [symbolic = %Self (constants.%Self.8196)]
// CHECK:STDOUT:         %Self.as_type.loc16_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.dc3d)]
// CHECK:STDOUT:         %.loc16_20.3: type = converted %Self.ref, %Self.as_type.loc16_20.2 [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.dc3d)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.dc3d) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Convert.2.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @Convert.2.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc16_32.1: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.4d84) = assoc_entity element0, %Convert.decl [symbolic = %assoc0.loc16_32.2 (constants.%assoc0.39c2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .Convert = %assoc0.loc16_32.1
// CHECK:STDOUT:     witness = (%Convert.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc19_6.2 as %Add.ref {
// CHECK:STDOUT:   %Op.decl: %Op.type.9ef5 = fn_decl @Op.2 [template = constants.%Op.11af] {
// CHECK:STDOUT:     %self.patt: %i32.builtin = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %i32.builtin = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %other.patt: %i32.builtin = binding_pattern other
// CHECK:STDOUT:     %other.param_patt: %i32.builtin = value_param_pattern %other.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc20_37: type = name_ref Self, @impl.1.%.loc19_6.2 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref.loc20_15: type = name_ref Self, @impl.1.%.loc19_6.2 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %i32.builtin = value_param runtime_param1
// CHECK:STDOUT:     %Self.ref.loc20_28: type = name_ref Self, @impl.1.%.loc19_6.2 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %other: %i32.builtin = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Op.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %.loc23_17.2 as %As.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.6e53 = fn_decl @Convert.3 [template = constants.%Convert.b6a5] {
// CHECK:STDOUT:     %self.patt: Core.IntLiteral = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: Core.IntLiteral = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc24_31.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc24_31.2: type = converted %int.make_type_signed, %.loc24_31.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.2.%.loc23_17.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %.loc27_17.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.9280 = fn_decl @Convert.4 [template = constants.%Convert.4e26] {
// CHECK:STDOUT:     %self.patt: Core.IntLiteral = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: Core.IntLiteral = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32.builtin = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32.builtin = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc28_31.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc28_31.2: type = converted %int.make_type_signed, %.loc28_31.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.3.%.loc27_17.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: %.loc31_6.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.5789 = fn_decl @Convert.5 [template = constants.%Convert.5d6f] {
// CHECK:STDOUT:     %self.patt: %i32.builtin = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %i32.builtin = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc32_42.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc32_42.2: type = converted %int_literal.make_type, %.loc32_42.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.4.%.loc31_6.2 [template = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(@Add.%Self: %Add.type) {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.d51f)]
// CHECK:STDOUT:   %Self.as_type.loc8_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.c8e7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7)](%other.param_patt: @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7)) -> @Op.1.%Self.as_type.loc8_15.1 (%Self.as_type.c8e7);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(@As.%T.loc11_14.1: type, @As.%Self.1: @As.%As.type (%As.type.8b04)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.8b04)]
// CHECK:STDOUT:   %Self: %As.type.8b04 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.a390)]
// CHECK:STDOUT:   %Self.as_type.loc12_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.5fac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type.loc12_20.1 (%Self.as_type.5fac)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(@ImplicitAs.%T.loc15_22.1: type, @ImplicitAs.%Self.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.a1f2)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.a1f2)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.a1f2 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.8196)]
// CHECK:STDOUT:   %Self.as_type.loc16_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.dc3d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type.loc16_20.1 (%Self.as_type.dc3d)]() -> @Convert.2.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self.param_patt: %i32.builtin](%other.param_patt: %i32.builtin) -> %i32.builtin = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: Core.IntLiteral]() -> %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: Core.IntLiteral]() -> %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: %i32.builtin]() -> Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.d51f) {
// CHECK:STDOUT:   %Self => constants.%Self.d51f
// CHECK:STDOUT:   %Self.as_type.loc8_15.1 => constants.%Self.as_type.c8e7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.a390) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.8b04
// CHECK:STDOUT:   %Self => constants.%Self.a390
// CHECK:STDOUT:   %Self.as_type.loc12_20.1 => constants.%Self.as_type.5fac
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%T.loc11_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc15_22.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc15_22.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%T, constants.%Self.8196) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.a1f2
// CHECK:STDOUT:   %Self => constants.%Self.8196
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => constants.%Self.as_type.dc3d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T.loc15_22.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Add.facet) {
// CHECK:STDOUT:   %Self => constants.%Add.facet
// CHECK:STDOUT:   %Self.as_type.loc8_15.1 => constants.%i32.builtin
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc11_14.2 => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt.loc11_14.2 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.8e7e
// CHECK:STDOUT:   %Self.2 => constants.%Self.a390
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.d050
// CHECK:STDOUT:   %Convert => constants.%Convert.5e5c
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.5961
// CHECK:STDOUT:   %assoc0.loc12_32.2 => constants.%assoc0.c6b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.8e7e
// CHECK:STDOUT:   %Self => constants.%As.facet
// CHECK:STDOUT:   %Self.as_type.loc12_20.1 => Core.IntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc15_22.2 => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt.loc15_22.2 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9821
// CHECK:STDOUT:   %Self.2 => constants.%Self.8196
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.c1fa
// CHECK:STDOUT:   %Convert => constants.%Convert.3381
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.9185
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.e51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%i32.builtin, constants.%ImplicitAs.facet.f011) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9821
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.f011
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => Core.IntLiteral
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc15_22.2 => Core.IntLiteral
// CHECK:STDOUT:   %T.patt.loc15_22.2 => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5f72
// CHECK:STDOUT:   %Self.2 => constants.%Self.8196
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.ddef
// CHECK:STDOUT:   %Convert => constants.%Convert.254f
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.6c13
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.143a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(Core.IntLiteral, constants.%ImplicitAs.facet.f95c) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5f72
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.f95c
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => constants.%i32.builtin
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_1.5b81: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %As.type.3d3f: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.24cd: %As.type.3d3f = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.1e6c: type = fn_type @Convert.1, @As(%T) [symbolic]
// CHECK:STDOUT:   %Convert.4bc0: %Convert.type.1e6c = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.414c: type = facet_access_type %Self.24cd [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.e0db: type = assoc_entity_type %As.type.3d3f, %Convert.type.1e6c [symbolic]
// CHECK:STDOUT:   %assoc0.cb697b.1: %Convert.assoc_type.e0db = assoc_entity element0, imports.%import_ref.9c35ee.1 [symbolic]
// CHECK:STDOUT:   %As.type.878f: type = facet_type <@As, @As(%i32.builtin)> [template]
// CHECK:STDOUT:   %Convert.type.1d52: type = fn_type @Convert.1, @As(%i32.builtin) [template]
// CHECK:STDOUT:   %Convert.3b8c: %Convert.type.1d52 = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.bb46: type = assoc_entity_type %As.type.878f, %Convert.type.1d52 [template]
// CHECK:STDOUT:   %assoc0.c563: %Convert.assoc_type.bb46 = assoc_entity element0, imports.%import_ref.9c35ee.1 [template]
// CHECK:STDOUT:   %assoc0.cb697b.2: %Convert.assoc_type.e0db = assoc_entity element0, imports.%import_ref.9c35ee.2 [symbolic]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [template]
// CHECK:STDOUT:   %Self.eeaa: %Add.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d331: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.400c: %ImplicitAs.type.d331 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.51bb: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [template]
// CHECK:STDOUT:   %Convert.type.6083: type = fn_type @Convert.2, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %Convert.de6e: %Convert.type.6083 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.9510: type = facet_access_type %Self.400c [symbolic]
// CHECK:STDOUT:   %Convert.assoc_type.407c: type = assoc_entity_type %ImplicitAs.type.d331, %Convert.type.6083 [symbolic]
// CHECK:STDOUT:   %assoc0.317cac.1: %Convert.assoc_type.407c = assoc_entity element0, imports.%import_ref.dbaf9d.1 [symbolic]
// CHECK:STDOUT:   %Convert.type.2a0b: type = fn_type @Convert.2, @ImplicitAs(%i32.builtin) [template]
// CHECK:STDOUT:   %Convert.7569: %Convert.type.2a0b = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.f84a: type = assoc_entity_type %ImplicitAs.type.51bb, %Convert.type.2a0b [template]
// CHECK:STDOUT:   %assoc0.695b: %Convert.assoc_type.f84a = assoc_entity element0, imports.%import_ref.dbaf9d.2 [template]
// CHECK:STDOUT:   %ImplicitAs.type.38ac: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.a9bd: type = fn_type @Convert.2, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.6c00: %Convert.type.a9bd = struct_value () [template]
// CHECK:STDOUT:   %Convert.assoc_type.bc85: type = assoc_entity_type %ImplicitAs.type.38ac, %Convert.type.a9bd [template]
// CHECK:STDOUT:   %assoc0.1227: %Convert.assoc_type.bc85 = assoc_entity element0, imports.%import_ref.dbaf9d.3 [template]
// CHECK:STDOUT:   %impl_witness.ffda: <witness> = impl_witness (imports.%import_ref.b92e) [template]
// CHECK:STDOUT:   %Convert.type.c68a: type = fn_type @Convert.3 [template]
// CHECK:STDOUT:   %Convert.fef7: %Convert.type.c68a = struct_value () [template]
// CHECK:STDOUT:   %int_2.ecc8: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Op.type.88c0: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Self.as_type.6ffc: type = facet_access_type %Self.eeaa [symbolic]
// CHECK:STDOUT:   %Op.assoc_type: type = assoc_entity_type %Add.type, %Op.type.88c0 [template]
// CHECK:STDOUT:   %assoc0.cee0: %Op.assoc_type = assoc_entity element0, imports.%import_ref.3bff [template]
// CHECK:STDOUT:   %impl_witness.724f: <witness> = impl_witness (imports.%import_ref.140d) [template]
// CHECK:STDOUT:   %Op.type.5456: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.0aa8: %Op.type.5456 = struct_value () [template]
// CHECK:STDOUT:   %int_3.a0f9: %i32.builtin = int_value 3 [template]
// CHECK:STDOUT:   %assoc0.317cac.2: %Convert.assoc_type.407c = assoc_entity element0, imports.%import_ref.dbaf9d.4 [symbolic]
// CHECK:STDOUT:   %impl_witness.5883: <witness> = impl_witness (imports.%import_ref.055c) [template]
// CHECK:STDOUT:   %int_3.1baa: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1baa, %i32.builtin [template]
// CHECK:STDOUT:   %int_4.0c1b: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.d532: <bound method> = bound_method %int_3.1baa, %Convert.fef7 [template]
// CHECK:STDOUT:   %Convert.bound.0268: <bound method> = bound_method %int_4.0c1b, %Convert.fef7 [template]
// CHECK:STDOUT:   %int_4.4f17: %i32.builtin = int_value 4 [template]
// CHECK:STDOUT:   %Op.bound.970f: <bound method> = bound_method %int_3.a0f9, %Op.0aa8 [template]
// CHECK:STDOUT:   %int_7: %i32.builtin = int_value 7 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, %i32.builtin) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %impl_witness.25bc: <witness> = impl_witness (imports.%import_ref.02a3) [template]
// CHECK:STDOUT:   %Convert.type.c8d6: type = fn_type @Convert.5 [template]
// CHECK:STDOUT:   %Convert.7180: %Convert.type.c8d6 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.2272: <bound method> = bound_method %int_3.1baa, %Convert.7180 [template]
// CHECK:STDOUT:   %Convert.bound.5039: <bound method> = bound_method %int_4.0c1b, %Convert.7180 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_3.a0f9, %int_4.4f17, %int_7) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.71ad
// CHECK:STDOUT:     .As = %import_ref.2f71
// CHECK:STDOUT:     .Add = %import_ref.4b51
// CHECK:STDOUT:     .ImplicitAs = %import_ref.7f17
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7a0c = import_ref Core//default, inst85 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.0012: @As.%Convert.assoc_type (%Convert.assoc_type.e0db) = import_ref Core//default, loc12_32, loaded [symbolic = @As.%assoc0 (constants.%assoc0.cb697b.2)]
// CHECK:STDOUT:   %import_ref.8460 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %import_ref.9c35ee.1 = import_ref Core//default, loc12_32, unloaded
// CHECK:STDOUT:   %import_ref.e3f8: <witness> = import_ref Core//default, loc19_17, loaded [template = constants.%impl_witness.724f]
// CHECK:STDOUT:   %import_ref.72a2 = import_ref Core//default, inst39 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.4d29: %Op.assoc_type = import_ref Core//default, loc8_41, loaded [template = constants.%assoc0.cee0]
// CHECK:STDOUT:   %import_ref.b752 = import_ref Core//default, Op, unloaded
// CHECK:STDOUT:   %import_ref.c8c7cd.1: type = import_ref Core//default, loc19_6, loaded [template = constants.%i32.builtin]
// CHECK:STDOUT:   %import_ref.d379: type = import_ref Core//default, loc19_13, loaded [template = constants.%Add.type]
// CHECK:STDOUT:   %import_ref.b30a: <witness> = import_ref Core//default, loc23_30, loaded [template = constants.%impl_witness.ffda]
// CHECK:STDOUT:   %import_ref.afb013.1: type = import_ref Core//default, loc23_17, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.d093: type = import_ref Core//default, loc23_28, loaded [template = constants.%As.type.878f]
// CHECK:STDOUT:   %import_ref.4ae0: <witness> = import_ref Core//default, loc27_38, loaded [template = constants.%impl_witness.25bc]
// CHECK:STDOUT:   %import_ref.01e7 = import_ref Core//default, inst128 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.86db: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.407c) = import_ref Core//default, loc16_32, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.317cac.2)]
// CHECK:STDOUT:   %import_ref.0389 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %import_ref.afb013.2: type = import_ref Core//default, loc27_17, loaded [template = Core.IntLiteral]
// CHECK:STDOUT:   %import_ref.d934: type = import_ref Core//default, loc27_36, loaded [template = constants.%ImplicitAs.type.51bb]
// CHECK:STDOUT:   %import_ref.dbaf9d.1 = import_ref Core//default, loc16_32, unloaded
// CHECK:STDOUT:   %import_ref.a1ef: <witness> = import_ref Core//default, loc31_38, loaded [template = constants.%impl_witness.5883]
// CHECK:STDOUT:   %import_ref.c8c7cd.2: type = import_ref Core//default, loc31_6, loaded [template = constants.%i32.builtin]
// CHECK:STDOUT:   %import_ref.4e3f: type = import_ref Core//default, loc31_36, loaded [template = constants.%ImplicitAs.type.38ac]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %arr.var: ref %array_type = var arr
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(constants.%T: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.3d3f)]
// CHECK:STDOUT:   %Self: %As.type.3d3f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.24cd)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.1, @As(%T) [symbolic = %Convert.type (constants.%Convert.type.1e6c)]
// CHECK:STDOUT:   %Convert: @As.%Convert.type (%Convert.type.1e6c) = struct_value () [symbolic = %Convert (constants.%Convert.4bc0)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @As.%As.type (%As.type.3d3f), @As.%Convert.type (%Convert.type.1e6c) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.e0db)]
// CHECK:STDOUT:   %assoc0: @As.%Convert.assoc_type (%Convert.assoc_type.e0db) = assoc_entity element0, imports.%import_ref.9c35ee.1 [symbolic = %assoc0 (constants.%assoc0.cb697b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.7a0c
// CHECK:STDOUT:     .Convert = imports.%import_ref.0012
// CHECK:STDOUT:     witness = (imports.%import_ref.8460)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Add [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.72a2
// CHECK:STDOUT:   .Op = imports.%import_ref.4d29
// CHECK:STDOUT:   witness = (imports.%import_ref.b752)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(constants.%T: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d331)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d331 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.400c)]
// CHECK:STDOUT:   %Convert.type: type = fn_type @Convert.2, @ImplicitAs(%T) [symbolic = %Convert.type (constants.%Convert.type.6083)]
// CHECK:STDOUT:   %Convert: @ImplicitAs.%Convert.type (%Convert.type.6083) = struct_value () [symbolic = %Convert (constants.%Convert.de6e)]
// CHECK:STDOUT:   %Convert.assoc_type: type = assoc_entity_type @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.d331), @ImplicitAs.%Convert.type (%Convert.type.6083) [symbolic = %Convert.assoc_type (constants.%Convert.assoc_type.407c)]
// CHECK:STDOUT:   %assoc0: @ImplicitAs.%Convert.assoc_type (%Convert.assoc_type.407c) = assoc_entity element0, imports.%import_ref.dbaf9d.1 [symbolic = %assoc0 (constants.%assoc0.317cac.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.01e7
// CHECK:STDOUT:     .Convert = imports.%import_ref.86db
// CHECK:STDOUT:     witness = (imports.%import_ref.0389)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: imports.%import_ref.c8c7cd.1 as imports.%import_ref.d379 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.e3f8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: imports.%import_ref.afb013.1 as imports.%import_ref.d093 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.b30a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: imports.%import_ref.afb013.2 as imports.%import_ref.d934 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.4ae0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4: imports.%import_ref.c8c7cd.2 as imports.%import_ref.4e3f [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%import_ref.a1ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.1(constants.%T: type, constants.%Self.24cd: %As.type.3d3f) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.3d3f)]
// CHECK:STDOUT:   %Self: %As.type.3d3f = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.24cd)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.414c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.1.%Self.as_type (%Self.as_type.414c)]() -> @Convert.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Convert.2(constants.%T: type, constants.%Self.400c: %ImplicitAs.type.d331) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.d331)]
// CHECK:STDOUT:   %Self: %ImplicitAs.type.d331 = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.400c)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.9510)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Convert.2.%Self.as_type (%Self.as_type.9510)]() -> @Convert.2.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3[%self.param_patt: Core.IntLiteral]() -> %i32.builtin = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Op.1(constants.%Self.eeaa: %Add.type) [from "core.carbon"] {
// CHECK:STDOUT:   %Self: %Add.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.eeaa)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.6ffc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @Op.1.%Self.as_type (%Self.as_type.6ffc)](%other.param_patt: @Op.1.%Self.as_type (%Self.as_type.6ffc)) -> @Op.1.%Self.as_type (%Self.as_type.6ffc);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op.2[%self.param_patt: %i32.builtin](%other.param_patt: %i32.builtin) -> %i32.builtin = "int.sadd" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4[%self.param_patt: %i32.builtin]() -> Core.IntLiteral = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5[%self.param_patt: Core.IntLiteral]() -> %i32.builtin = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_3.loc4_44: Core.IntLiteral = int_value 3 [template = constants.%int_3.1baa]
// CHECK:STDOUT:   %int_4.loc4_47: Core.IntLiteral = int_value 4 [template = constants.%int_4.0c1b]
// CHECK:STDOUT:   %int_3.loc4_51: Core.IntLiteral = int_value 3 [template = constants.%int_3.1baa]
// CHECK:STDOUT:   %int_32.loc4_56: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc4_56: init type = call constants.%Int(%int_32.loc4_56) [template = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_56.1: type = value_of_initializer %int.make_type_signed.loc4_56 [template = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_56.2: type = converted %int.make_type_signed.loc4_56, %.loc4_56.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_53: %Convert.type.1d52 = impl_witness_access constants.%impl_witness.ffda, element0 [template = constants.%Convert.fef7]
// CHECK:STDOUT:   %Convert.bound.loc4_53: <bound method> = bound_method %int_3.loc4_51, %impl.elem0.loc4_53 [template = constants.%Convert.bound.d532]
// CHECK:STDOUT:   %int.convert_checked.loc4_53: init %i32.builtin = call %Convert.bound.loc4_53(%int_3.loc4_51) [template = constants.%int_3.a0f9]
// CHECK:STDOUT:   %.loc4_53.1: %i32.builtin = value_of_initializer %int.convert_checked.loc4_53 [template = constants.%int_3.a0f9]
// CHECK:STDOUT:   %.loc4_53.2: %i32.builtin = converted %int_3.loc4_51, %.loc4_53.1 [template = constants.%int_3.a0f9]
// CHECK:STDOUT:   %int_4.loc4_64: Core.IntLiteral = int_value 4 [template = constants.%int_4.0c1b]
// CHECK:STDOUT:   %int_32.loc4_69: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc4_69: init type = call constants.%Int(%int_32.loc4_69) [template = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_69.1: type = value_of_initializer %int.make_type_signed.loc4_69 [template = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc4_69.2: type = converted %int.make_type_signed.loc4_69, %.loc4_69.1 [template = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_66: %Convert.type.1d52 = impl_witness_access constants.%impl_witness.ffda, element0 [template = constants.%Convert.fef7]
// CHECK:STDOUT:   %Convert.bound.loc4_66: <bound method> = bound_method %int_4.loc4_64, %impl.elem0.loc4_66 [template = constants.%Convert.bound.0268]
// CHECK:STDOUT:   %int.convert_checked.loc4_66: init %i32.builtin = call %Convert.bound.loc4_66(%int_4.loc4_64) [template = constants.%int_4.4f17]
// CHECK:STDOUT:   %.loc4_66.1: %i32.builtin = value_of_initializer %int.convert_checked.loc4_66 [template = constants.%int_4.4f17]
// CHECK:STDOUT:   %.loc4_66.2: %i32.builtin = converted %int_4.loc4_64, %.loc4_66.1 [template = constants.%int_4.4f17]
// CHECK:STDOUT:   %impl.elem0.loc4_61: %Op.type.88c0 = impl_witness_access constants.%impl_witness.724f, element0 [template = constants.%Op.0aa8]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc4_53.2, %impl.elem0.loc4_61 [template = constants.%Op.bound.970f]
// CHECK:STDOUT:   %int.sadd: init %i32.builtin = call %Op.bound(%.loc4_53.2, %.loc4_66.2) [template = constants.%int_7]
// CHECK:STDOUT:   %.loc4_73.1: %tuple.type = tuple_literal (%int_3.loc4_44, %int_4.loc4_47, %int.sadd)
// CHECK:STDOUT:   %impl.elem0.loc4_73.1: %Convert.type.2a0b = impl_witness_access constants.%impl_witness.25bc, element0 [template = constants.%Convert.7180]
// CHECK:STDOUT:   %Convert.bound.loc4_73.1: <bound method> = bound_method %int_3.loc4_44, %impl.elem0.loc4_73.1 [template = constants.%Convert.bound.2272]
// CHECK:STDOUT:   %int.convert_checked.loc4_73.1: init %i32.builtin = call %Convert.bound.loc4_73.1(%int_3.loc4_44) [template = constants.%int_3.a0f9]
// CHECK:STDOUT:   %.loc4_73.2: init %i32.builtin = converted %int_3.loc4_44, %int.convert_checked.loc4_73.1 [template = constants.%int_3.a0f9]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc4_73.3: ref %i32.builtin = array_index file.%arr.var, %int_0
// CHECK:STDOUT:   %.loc4_73.4: init %i32.builtin = initialize_from %.loc4_73.2 to %.loc4_73.3 [template = constants.%int_3.a0f9]
// CHECK:STDOUT:   %impl.elem0.loc4_73.2: %Convert.type.2a0b = impl_witness_access constants.%impl_witness.25bc, element0 [template = constants.%Convert.7180]
// CHECK:STDOUT:   %Convert.bound.loc4_73.2: <bound method> = bound_method %int_4.loc4_47, %impl.elem0.loc4_73.2 [template = constants.%Convert.bound.5039]
// CHECK:STDOUT:   %int.convert_checked.loc4_73.2: init %i32.builtin = call %Convert.bound.loc4_73.2(%int_4.loc4_47) [template = constants.%int_4.4f17]
// CHECK:STDOUT:   %.loc4_73.5: init %i32.builtin = converted %int_4.loc4_47, %int.convert_checked.loc4_73.2 [template = constants.%int_4.4f17]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b81]
// CHECK:STDOUT:   %.loc4_73.6: ref %i32.builtin = array_index file.%arr.var, %int_1
// CHECK:STDOUT:   %.loc4_73.7: init %i32.builtin = initialize_from %.loc4_73.5 to %.loc4_73.6 [template = constants.%int_4.4f17]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc8]
// CHECK:STDOUT:   %.loc4_73.8: ref %i32.builtin = array_index file.%arr.var, %int_2
// CHECK:STDOUT:   %.loc4_73.9: init %i32.builtin = initialize_from %int.sadd to %.loc4_73.8 [template = constants.%int_7]
// CHECK:STDOUT:   %.loc4_73.10: init %array_type = array_init (%.loc4_73.4, %.loc4_73.7, %.loc4_73.9) to file.%arr.var [template = constants.%array]
// CHECK:STDOUT:   %.loc4_74: init %array_type = converted %.loc4_73.1, %.loc4_73.10 [template = constants.%array]
// CHECK:STDOUT:   assign file.%arr.var, %.loc4_74
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(@Convert.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.1(constants.%T, constants.%Self.24cd) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.3d3f
// CHECK:STDOUT:   %Self => constants.%Self.24cd
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.414c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.878f
// CHECK:STDOUT:   %Self => constants.%Self.24cd
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.1d52
// CHECK:STDOUT:   %Convert => constants.%Convert.3b8c
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.bb46
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c563
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %T.patt => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.51bb
// CHECK:STDOUT:   %Self => constants.%Self.400c
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.2a0b
// CHECK:STDOUT:   %Convert => constants.%Convert.7569
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.f84a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.695b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(@Convert.2.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Convert.2(constants.%T, constants.%Self.400c) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.d331
// CHECK:STDOUT:   %Self => constants.%Self.400c
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.9510
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %T.patt => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.38ac
// CHECK:STDOUT:   %Self => constants.%Self.400c
// CHECK:STDOUT:   %Convert.type => constants.%Convert.type.a9bd
// CHECK:STDOUT:   %Convert => constants.%Convert.6c00
// CHECK:STDOUT:   %Convert.assoc_type => constants.%Convert.assoc_type.bc85
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Op.1(constants.%Self.eeaa) {
// CHECK:STDOUT:   %Self => constants.%Self.eeaa
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.6ffc
// CHECK:STDOUT: }
// CHECK:STDOUT:
